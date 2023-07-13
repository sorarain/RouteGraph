import json
import os
import argparse
from time import time
from typing import List, Dict, Any
from functools import reduce

import numpy as np
import dgl

import torch
import torch.nn as nn

from data.load_data import load_data
from model.RouteGNN import NetlistGNN
# from log.store_scatter import store_scatter
# from utils.output import printout_xf1
from train.train_wirelength import train_wirelength

import warnings

argparser = argparse.ArgumentParser("Training")

argparser.add_argument('--name', type=str, default='main')
argparser.add_argument('--test', type=str, default='superblue19')
argparser.add_argument('--epochs', type=int, default=20)
argparser.add_argument('--train_epoch', type=int, default=5)
argparser.add_argument('--batch', type=int, default=1)
argparser.add_argument('--lr', type=float, default=2e-4)
argparser.add_argument('--weight_decay', type=float, default=1e-5)
argparser.add_argument('--lr_decay', type=float, default=2e-2)
argparser.add_argument('--beta', type=float, default=0.5)

argparser.add_argument('--app_name', type=str, default='')
argparser.add_argument('--win_x', type=float, default=32)
argparser.add_argument('--win_y', type=float, default=40)
argparser.add_argument('--win_cap', type=int, default=5)

argparser.add_argument('--model', type=str, default='')  # ''
argparser.add_argument('--trans', type=bool, default=False)  # ''
argparser.add_argument('--layers', type=int, default=3)  # 3
argparser.add_argument('--node_feats', type=int, default=64)  # 64
argparser.add_argument('--net_feats', type=int, default=128)  # 128
argparser.add_argument('--pin_feats', type=int, default=16)  # 16
argparser.add_argument('--hanna_feats', type=int, default=4)  # 4
argparser.add_argument('--topo_geom', type=str, default='both')  # default
argparser.add_argument('--recurrent', type=bool, default=False)  # False
argparser.add_argument('--topo_conv_type', type=str, default='CFCNN')  # CFCNN
argparser.add_argument('--pos_code', type=float, default=0.0)  # 0.0

argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--device', type=str, default='cuda:0')
argparser.add_argument('--hashcode', type=str, default='100000')
argparser.add_argument('--idx', type=int, default=8)
argparser.add_argument('--itermax', type=int, default=2500)
argparser.add_argument('--scalefac', type=float, default=7.0)
argparser.add_argument('--outtype', type=str, default='tanh')
argparser.add_argument('--binx', type=int, default=32)
argparser.add_argument('--biny', type=int, default=40)

argparser.add_argument('--graph_scale', type=int, default=10000)
args = argparser.parse_args()
MODEL_DIR="./param"
LOG_DIR=f"./log/hpwl-{args.test}"
FIG_DIR='./log/hpwl-temp'

train_netlists_names=[
    # f'{os.path.abspath(".")}/data/data/superblue2',
    # f'{os.path.abspath(".")}/data/data/superblue3',
    # f'{os.path.abspath(".")}/data/data/superblue6',
    # f'{os.path.abspath(".")}/data/data/superblue7',
    # f'{os.path.abspath(".")}/data/data/superblue9',
    # f'{os.path.abspath(".")}/data/data/superblue11',
    # f'{os.path.abspath(".")}/data/data/superblue14',
    f'{os.path.abspath(".")}/data/data/superblue19',
    ]
validation_netlists_names=[
                        # f'{os.path.abspath(".")}/data/data/superblue16',
                        f'{os.path.abspath(".")}/data/data/superblue19',
                        ]
test_netlists_names=[
                    f'{os.path.abspath(".")}/data/data/superblue19',
                    ]

if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
if not os.path.isdir(FIG_DIR):
    os.mkdir(FIG_DIR)

train_wirelength(args,
                train_netlists_names=train_netlists_names,
                validation_netlists_names=validation_netlists_names,
                test_netlists_names=test_netlists_names,
                log_dir=LOG_DIR,
                fig_dir=FIG_DIR,
                model_dir=MODEL_DIR
                )