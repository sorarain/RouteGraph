import json
import os
from time import time
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
    with open('../data/graph.pickle', 'rb') as f:
        train_list_netlist = pickle.load(f)

    print('###MODEL###')
    #model feature sizes
    # node= cell
    in_node_feats = train_list_netlist[0][1].nodes['cell'].data['hv'].shape[1]
    in_net_feats = train_list_netlist[0][1].nodes['net'].data['hv'].shape[1]
    in_hanna_feats = train_list_netlist[1][1].nodes['hanna'].data['hv'].shape[1]
    in_pin_feats = train_list_netlist[0][1].edges['pinned'].data['feats'].shape[1]

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
    def train(ltg):
        if args.trans:
            for p in model.net_readout_params:
                p.train()
        else:
            model.train()
        t1 = time()
        losses = []
        n_tuples = len(train_list_netlist)
        for i, (hetero_graph, hanna_graph) in enumerate(ltg):
            hetreo_graph, hanna_graph = to_device(hetero_graph,hanna_graph)
            optimizer.zero_grad()
            pred_cell, pred_net = forward(hanna_graph)
            cell_label = hetero_graph.nodes['cell'].data['label'].to(device)
            net_label = hetero_graph.nodes['cell'].data['label'].to(device)
            cell_loss =loss_f(pred_cell, cell_label.float())
            net_loss = loss_f(pred_net, net_label.float())
            loss = cell_loss + net_loss
            print(loss)
            losses.append(loss)
            if len(losses) >= args.batch or i == n_tuples - 1:
                sum(losses).backward()
                optimizer.step()
                losses.clear()
        scheduler.step()
        print(f"\tTraining time per epoch: {time() - t1}")

    t0 = time()
    for _ in range(args.train_epoch):
        train(train_list_netlist)
    logs[-1].update({'train_time': time() - t0})










