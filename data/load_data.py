import os
import os.path.join as osp
import sys

import numpy as np
import itertools
import dgl
import pickle
import tqdm
import pandas as pd
from queue import Queue
import torch
import time

from data.graph import load_graph

def load_data(netlist_dir:str,save_type:int=1):
    if save_type == 1 and os.path.exists(osp(netlist_dir,'graph.pickle')):
        with open(osp(netlist_dir,'graph.pickle'),"rb") as f:
            list_tuple_graph = pickle.load(f)
            return list_tuple_graph
    list_tuple_graph = load_graph(netlist_dir)
    with open(osp(netlist_dir,'graph.pickle'),"wb") as f:
        pickle.dump(list_tuple_graph, f)
    return list_tuple_graph
