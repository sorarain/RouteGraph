import json
import os
from time import time

import numpy
import pandas
from typing import List, Dict, Any, Tuple
from functools import reduce
import pickle
import numpy as np
import dgl
import argparse
import torch
import torch.nn as nn

from data.load_data import load_data
from model.RouteGNN import NetlistGNN
from utils.output import printout, get_grid_level_corr, mean_dict


def train_congestion(
        args,
        train_dataset_names=None,
        validation_dataset_names = None,
        test_dataset_names = None,
        log_dir = None,
        fig_dir = None,
        model_dir = None,):
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
        'HANNA_FEATS': args.hanna_feats,
    }

    # load data
    # train_list_netlist = load_data(train_dataset_names)
    # validation_list_netlist = load_data(validation_dataset_names)
    # test_list_netlist = load_data(test_dataset_names)

    #temporory training/ testing set
    with open('../data/graph.pickle', 'rb') as f:
        datalist = pickle.load(f)
    train_list_netlist = [datalist[:50]]
    test_list_netlist = [datalist[51:60]]


    print('###MODEL###')
    #model feature sizes
    # node= cell
    in_node_feats = train_list_netlist[0][0][1].nodes['cell'].data['hv'].shape[1]
    in_net_feats = train_list_netlist[0][0][1].nodes['net'].data['hv'].shape[1]
    in_hanna_feats = train_list_netlist[0][1][1].nodes['hanna'].data['hv'].shape[1]
    in_pin_feats = train_list_netlist[0][0][1].edges['pinned'].data['feats'].shape[1]

    model = NetlistGNN(
        in_node_feats=in_node_feats,
        in_net_feats=in_net_feats,
        in_hanna_feats=in_hanna_feats,
        in_pin_feats=in_pin_feats,
        config=config,
        n_target=1,
    ).to(device)
    #load model
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=(1 - args.lr_decay))
    if args.beta < 1e-5:
        print(f'### USE L1Loss ###')
        loss_f = nn.L1Loss()
    elif args.beta > 7.0:
        print(f'### USE MSELoss ###')
        loss_f = nn.MSELoss()
    else:
        print(f'### USE SmoothL1Loss with beta={args.beta} ###')
        loss_f = nn.SmoothL1Loss(beta=args.beta)

    best_rmse = 1e8

    def to_device(a,b):
        return a.to(device), b.to(device)
    def forward(hanna_graph):
        in_node_feat = hanna_graph.nodes['cell'].data['hv']
        in_net_feat = hanna_graph.nodes['net'].data['hv']
        in_pin_feat = hanna_graph.edges['pinned'].data['feats']
        in_hanna_feat = hanna_graph.nodes['hanna'].data['hv']
        pred_cell, pred_net = model.forward(in_node_feat=in_node_feat,in_net_feat=in_net_feat,
                                in_pin_feat=in_pin_feat,in_hanna_feat=in_hanna_feat,node_net_graph=hanna_graph)
        if args.scalefac:
            pred_cell = pred_cell * args.scalefac
            pred_net = pred_net * args.scalefac
        return pred_cell, pred_net

    #training
    def train(ltgs:List[List[Tuple[dgl.DGLHeteroGraph, dgl.DGLHeteroGraph]]]):
        ltg = []
        for ltg_ in ltgs:
            ltg.extend(ltg_)
        if args.trans:
            for p in model.net_readout_params:
                p.train()
        else:
            model.train()
        t1 = time()
        losses = []
        n_tuples = len(ltg)
        for i, (hetero_graph, hanna_graph) in enumerate(ltg):
            hetreo_graph, hanna_graph = to_device(hetero_graph,hanna_graph)
            optimizer.zero_grad()
            cell_pred, net_pred = forward(hanna_graph)
            cell_label = hetero_graph.nodes['cell'].data['label'].to(device)
            #net_label = hetero_graph.nodes['net'].data['label'].to(device)
            cell_loss =loss_f(cell_pred, cell_label.float())
            #net_loss = loss_f(net_pred, net_label.float())
            #loss = cell_loss + net_loss
            loss = cell_loss
            print(loss)
            losses.append(loss)
            if len(losses) >= args.batch or i == n_tuples - 1:
                sum(losses).backward()
                optimizer.step()
                losses.clear()
        scheduler.step()
        print(f"\tTraining time per epoch: {time() - t1}")

    def evaluate(ltgs: List[List[Tuple[dgl.DGLHeteroGraph, dgl.DGLHeteroGraph]]], set_name:str,
                 explicit_names:List[str], verbose =True):
        model.eval()
        ds =[]
        for data_name, ltg in zip(explicit_names, ltgs):
            n_node = sum(map(lambda x: x[0].number_of_nodes(), ltg))
            outputdata = np.zeros((n_node, 5))
            p = 0
            for i, (hetero_graph, hanna_graph) in enumerate(ltg):
                hetero_graph, hanna_graph = to_device(hetero_graph, hanna_graph)
                cell_label = hetero_graph.nodes['cell'].data['label'].cpu().data.numpy().flatten()
                ln = len(cell_label)
                #net_label = hetero_graph.nodes['net'].data['label'].cpu().data.numpy().flatten()
                cell_pred, net_pred = forward(hanna_graph)
                cell_pred = cell_pred.cpu().data.numpy().flatten()
                #net_pred = net_pred.cpu().data.numpy().flatten()
                #print(hetero_graph.nodes['cell'].data)
                density = hetero_graph.nodes['cell'].data['hv'][:,6].cpu().data.numpy()
                output_pos = (hetero_graph.nodes['cell'].data['pos'].cpu().data.numpy())
                #label = numpy.concatenate(cell_label,net_label)
                #pred = numpy.concatenate(cell_pred,net_pred)

                outputdata[p:p+ln,0], outputdata[p:p+ln,1] = cell_label, cell_pred
                outputdata[p:p+ln,2:4], outputdata[p:p+ln, 4] = output_pos, density
                p += ln
            outputdata = outputdata[:p, :]
        if args.topo_geom != 'topo':
            bad_node = outputdata[:, 4] < 0.5
            outputdata[bad_node, 1] = outputdata[bad_node, 0]
        print(f'\t{data_name}:')
        d = printout(outputdata[:, 0], outputdata[:, 1], "\t\tNODE_LEVEL: ", f'{set_name}node_level_',
                     verbose=verbose)
        # save model
        if model_dir is not None and set_name == 'validate_':
            rmse = d[f'{set_name}node_level_rmse']
            nonlocal best_rmse
            if rmse < best_rmse:
                best_rmse = rmse
                print(f'\tSaving model to {model_dir}/{args.name}.pkl ...:')
                torch.save(model.state_dict(), f'{model_dir}/{args.name}.pkl')
        d1, d2 = get_grid_level_corr(outputdata[:, :4], args.binx, args.biny,
                                     int(np.rint(np.max(outputdata[:, 2]) / args.binx)) + 1,
                                     int(np.rint(np.max(outputdata[:, 3]) / args.biny)) + 1,
                                     set_name=set_name, verbose=verbose)
        d.update(d1)
        d.update(d2)
        ds.append(d)
        logs[-1].update(mean_dict(ds))

    train_dataset_names = ['hello']
    test_dataset_names = ['olleh']
    for epoch in range(0, args.epochs + 1):
        print(f'##### EPOCH {epoch} #####')
        print(f'\tLearning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        logs.append({'epoch': epoch})
        t0 = time()
        if epoch:
            for _ in range(args.train_epoch):
                train(train_list_netlist)
        logs[-1].update({'train_time': time() - t0})
        t2  = time()
        evaluate(train_list_netlist, 'train_', train_dataset_names, verbose=False)
        if len(test_dataset_names):
            evaluate(test_list_netlist, 'test_', test_dataset_names)
        print("\tinference time", time() - t2)
        logs[-1].update({'eval_time': time() - t2})
        if log_dir is not None:
            with open(f'{log_dir}/{args.name}.json', 'w+') as fp:
                json.dump(logs, fp)








