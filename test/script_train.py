import torch
import dgl
from data.graph import load_data
from typing import List, Dict, Any, Tuple

def train(
    args,
    train_dataset_names: List[str],
    validate_dataset_names: List[str],
    test_dataset_names: List[str],
    log_dir: str = None,
    fig_dir: str = None,
    model_dir: str = None,
    ):

    logs: List[Dict[str, Any]] = []
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(args.device)
    if not args.device == 'cpu':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)

    config = {
        'N_LAYER': args.layers,
        'NODE_FEATS': args.node_feats,
        'NET_FEATS': args.net_feats,
        'PIN_FEATS': args.pin_feats,
        'EDGE_FEATS': args.edge_feats,
    }

    def get_list_tuple_graph_from_dataset_name(dataset_name) -> List[Tuple[dgl.DGLGraph, dgl.DGLHeteroGraph]]:
        return load_data(f'/root/autodl-tmp/{dataset_name}_processed')

    train_list_netlist = [get_list_tuple_graph_from_dataset_name(dataset_name) for dataset_name in train_dataset_names]
    validate_list_netlist = [get_list_tuple_graph_from_dataset_name(dataset_name) for dataset_name in validate_dataset_names]
    test_list_netlist = [get_list_tuple_graph_from_dataset_name(dataset_name) for dataset_name in test_dataset_names]

    print('##### MODEL #####')
    in_node_feats = train_list_netlist[0][0][1].nodes['node'].data['hv'].shape[1]
    in_net_feats = train_list_netlist[0][0][1].nodes['net'].data['hv'].shape[1]
    in_pin_feats = train_list_netlist[0][0][1].edges['pinned'].data['he'].shape[1]
    model = NetlistGNN(
        in_node_feats=in_node_feats,
        in_net_feats=in_net_feats,
        in_pin_feats=in_pin_feats,
        in_edge_feats=1,
        activation=args.outtype,
        config=config,
        agg_type=args.agg_type,
    ).to(device)
    # model reuse
    if args.model:
        model_dicts = torch.load(f'model/{args.model}.pkl', map_location=device)
        model.load_state_dict(model_dicts)
        model.eval()
    n_param = 0
    for name, param in model.named_parameters():
        print(f'\t{name}: {param.shape}')
        n_param += reduce(lambda x, y: x * y, param.shape)
    print(f'# of parameters: {n_param}')

    # optimizer
    loss_f = nn.SmoothL1Loss(beta=args.beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=(1 - args.lr_decay))
    
    best_rmse = 1e8

    for epoch in range(0, args.epochs + 1):
        print(f'##### EPOCH {epoch} #####')
        print(f'\tLearning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        logs.append({'epoch': epoch})

        def forward(homo_graph, hetero_graph):
            optimizer.zero_grad()
            in_node_feat = hetero_graph.nodes['node'].data['hv']
            in_net_feat = hetero_graph.nodes['net'].data['hv']
            pred, _ = model.forward(
                in_node_feat=in_node_feat,
                in_net_feat=in_net_feat,
                in_pin_feat=hetero_graph.edges['pinned'].data['he'],
                in_edge_feat=hetero_graph.edges['near'].data['he'],
                node_net_graph=hetero_graph,
            )
            return pred * args.scalefac

        def train(ltgs: List[List[Tuple[dgl.DGLGraph, dgl.DGLHeteroGraph]]]):
            ltg = []
            for ltg_ in ltgs:
                ltg.extend(ltg_)
            model.train()
            t1 = time()
            losses = []
            n_tuples = len(ltg)
            for j, (homo_graph, hetero_graph) in enumerate(ltg):
                homo_graph, hetero_graph = to_device(homo_graph, hetero_graph)
                pred = forward(homo_graph, hetero_graph)
                batch_labels = homo_graph.ndata['label']
                weight = 1 / hetero_graph.nodes['node'].data['hv'][:, 6]
                weight[torch.isinf(weight)] = 0.0
                if args.topo_geom != 'topo':
                    loss = torch.sum(((pred.view(-1) - batch_labels.float()) ** 2) * weight) / torch.sum(weight)
                else:
                    loss = loss_f(pred.view(-1), batch_labels.float())
                losses.append(loss)
                if len(losses) >= args.batch or j == n_tuples - 1:
                    sum(losses).backward()
                    optimizer.step()
                    losses.clear()
            scheduler.step()
            print(f"\tTraining time per epoch: {time() - t1}")

        def evaluate(ltgs: List[List[Tuple[dgl.DGLGraph, dgl.DGLHeteroGraph]]], set_name: str,
                     explicit_names: List[str], verbose=True):
            model.eval()
            print(f'\tEvaluate {set_name}:')
            ds = []
            for data_name, ltg in zip(explicit_names, ltgs):
                n_node = sum(map(lambda x: x[0].number_of_nodes(), ltg))
                outputdata = np.zeros((n_node, 5))
                p = 0
                for j, (homo_graph, hetero_graph) in enumerate(ltg):
                    homo_graph, hetero_graph = to_device(homo_graph, hetero_graph)
                    pred = forward(homo_graph, hetero_graph)
                    density = homo_graph.ndata['feat'][:, 6].cpu().data.numpy()
                    output_labels = homo_graph.ndata['label']
                    output_pos = (homo_graph.ndata['pos'].cpu().data.numpy())
                    output_predictions = pred
                    tgt = output_labels.cpu().data.numpy().flatten()
                    prd = output_predictions.cpu().data.numpy().flatten()
                    ln = len(tgt)
                    outputdata[p:p + ln, 0], outputdata[p:p + ln, 1] = tgt, prd
                    outputdata[p:p + ln, 2:4], outputdata[p:p + ln, 4] = output_pos, density
                    p += ln
                outputdata = outputdata[:p, :]
                if args.topo_geom != 'topo':
                    bad_node = outputdata[:, 4] < 0.5
                    outputdata[bad_node, 1] = outputdata[bad_node, 0]
                print(f'\t{data_name}:')
                d = printout(outputdata[:, 0], outputdata[:, 1], "\t\tNODE_LEVEL: ", f'{set_name}node_level_',
                             verbose=verbose)
                if model_dir is not None and set_name == 'validate_':
                    rmse = d[f'{set_name}node_level_rmse']
                    nonlocal best_rmse
                    if rmse < best_rmse:
                        best_rmse = rmse
                        print(f'\tSaving model to {model_dir}/{args.name}.pkl ...:')
                        torch.save(model.state_dict(), f'{model_dir}/{args.name}.pkl')

                d.update(d1)
                d.update(d2)
                ds.append(d)

            logs[-1].update(mean_dict(ds))

        t0 = time()
        if epoch:
            for _ in range(args.train_epoch):
                train(train_list_netlist)
        logs[-1].update({'train_time': time() - t0})
        t2 = time()
        evaluate(train_list_netlist, 'train_', train_dataset_names, verbose=False)
        if len(validate_list_netlist):
            evaluate(validate_list_netlist, 'validate_', validate_dataset_names)
        if len(test_dataset_names):
            evaluate(test_list_netlist, 'test_', test_dataset_names)
        # exit(123)
        print("\tinference time", time() - t2)
        logs[-1].update({'eval_time': time() - t2})
        if log_dir is not None:
            with open(f'{log_dir}/{args.name}.json', 'w+') as fp:
                json.dump(logs, fp)