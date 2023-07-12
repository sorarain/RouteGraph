import os
import os.path as osp
import sys

from data.graph import load_graph

sys.path.append(os.path.abspath("."))

import numpy as np
import itertools
import dgl
import pickle
import tqdm
import pandas as pd
from queue import Queue
import torch
import time


def load_data(netlist_dir:str,save_type:int=1):
    if save_type == 1 and os.path.exists(osp.join(netlist_dir,'graph.pickle')):
        with open(osp.join(netlist_dir,'graph.pickle'),"rb") as f:
            list_tuple_graph = pickle.load(f)
            return list_tuple_graph
    list_hetero_graph,list_route_graph = load_graph(netlist_dir)
    list_tuple_graph = list(zip(list_hetero_graph, list_route_graph))
    with open(osp.join(netlist_dir,'graph.pickle'),"wb") as f:
        pickle.dump(list_tuple_graph, f)
    return list_tuple_graph

if __name__ == '__main__':
    load_data("/root/autodl-tmp/data/superblue19",2)
