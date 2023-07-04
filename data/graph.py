import os
import sys
sys.path.append(os.path.join(os.path.abspath('.'),'build'))
sys.path.append(os.path.abspath('.'))

import PlaceDB
import Params
from partition_graph import partition_graph

import numpy as np
import itertools
import dgl
import pickle
import tqdm
import pandas as pd
from queue import Queue
import torch
import time

def get_w():
    param_dir_list = [
        'test/dac2012/superblue2.json',
        'test/dac2012/superblue3.json',
        'test/dac2012/superblue6.json',
        'test/dac2012/superblue7.json',
        'test/dac2012/superblue9.json',
        'test/dac2012/superblue11.json',
        'test/dac2012/superblue12.json',
        'test/dac2012/superblue14.json',
        'test/dac2012/superblue16.json',
        'test/dac2012/superblue19.json',
    ]
    netlist_name_list = [
        'superblue2',
        'superblue3',
        'superblue6',
        'superblue7',
        'superblue9',
        'superblue11',
        'superblue12',
        'superblue14',
        'superblue16',
        'superblue19',
    ]

    w_list = []
    netlist_info = []

    for param_dir,netlist_name in zip(param_dir_list,netlist_name_list):
        params = Params.Params()
        params.load(param_dir)
        placedb = PlaceDB.PlaceDB()
        placedb(params)
        num_pins = len(placedb.pin_offset_x)
        num_hanan_points = 0
        for net2pinid in placedb.net2pin_map:
            num_hanan_points += len(net2pinid) * len(net2pinid)
        print("-----------------")
        print(f"netlist {netlist_name}")
        print(f"num pins {num_pins}")
        print(f"num hanan points {num_hanan_points}")
        print(f"hanan points / pins {num_hanan_points / num_pins}")
        netlist_info.append([netlist_name,num_pins,num_hanan_points,num_hanan_points / num_pins])
        w_list.append(num_hanan_points / num_pins)
    print(w_list)
    print(np.mean(w_list))
    netlist_info = np.array(netlist_info)
    df = pd.DataFrame()
    df['netlist_name'] = netlist_info[:,0]
    df['num pins'] = netlist_info[:,1]
    df['num hanan points'] = netlist_info[:,2]
    df['hanan points / pins'] = netlist_info[:,3]
    df.to_excel("./info.xlsx",index=False)

def get_edge(netlist_dir):
    edge_dict = {
        0:[
            [0,20,20,0],
            [2,30,20,1],
            [3,10,20,1]
        ],
        1:[
            [3,20,20,0],
            [5,20,-20,1],
        ],
        2:[
            [1,20,20,0],
            [3,30,20,1],
            [4,20,20,1],
        ]
    }
    # with open(f"/root/autodl-tmp/{netlist_name}_processed/edge.pkl",'rb') as fp:
    with open(os.path.join(netlist_dir,"edge.pkl"),'rb') as fp:
        edge_dict = pickle.load(fp)
    return edge_dict
def get_pos(netlist_dir):
    node_pos = np.array([
        [100,200],
        [100,400],
        [200,100],
        [200,300],
        [200,500],
        [300,300],
    ])
    # node_pos_x = np.load(f"/root/autodl-tmp/{netlist_name}_processed/xdata_900.npy")
    # node_pos_y = np.load(f"/root/autodl-tmp/{netlist_name}_processed/ydata_900.npy")
    node_pos_x = np.load(os.path.join(netlist_dir,"xdata_900.npy"))
    node_pos_y = np.load(os.path.join(netlist_dir,"ydata_900.npy"))
    node_pos = np.vstack((node_pos_x,node_pos_y)).T
    return node_pos

def intersect(pos_x_1_1,pos_y_1_1,pos_x_2_1,pos_y_2_1,
            pos_x_1_2,pos_y_1_2,pos_x_2_2,pos_y_2_2):
    # pos_x_1_2,pos_y_1_2,pos_x_2_2,pos_y_2_2,_ = line_2_info
    if pos_x_1_1 == pos_x_2_1:
        return not(pos_y_1_1 > pos_y_2_2 or pos_y_2_1 < pos_y_1_2)
    if pos_y_1_1 == pos_y_2_1:
        return not(pos_x_1_1 > pos_x_2_2 or pos_x_2_1 < pos_x_1_2)
    else:
        assert 0==1,"error in intersect"

def build_group_hanna_point(pos_1_dict,hanna_point_info,max_pos_2,min_pos_2,id_num_hanna_points,key,
                            route_edge_us,route_edge_vs,pin_dict):#key=['x','y']
    pos_1_list = list(set(pos_1_dict.keys()))
    id2pinpos_y = set()
    for pos_1 in pos_1_list:
        for pos_2 in pos_1_dict[pos_1]:
            id2pinpos_y.add(pos_2)
    id2pinpos_y = list(id2pinpos_y)
    id2pinpos_y.sort()
    pinpos_y2id = {}
    for id,pinpos_y in enumerate(id2pinpos_y):
        pinpos_y2id[pinpos_y] = id
    # print("----------------")
    # print(min_pos_2,max_pos_2)
    # print(pin_dict)
    # print(pos_1_dict)
    pos_1_list.sort()
    tmp_pre_hanna_point_list = []
    for i,pos_1 in enumerate(pos_1_list):
        #########create group hanna point###########
        pos_2_list = list(set(pos_1_dict[pos_1].copy()))
        pos_2_list.sort()
        pre_pos_2 = min_pos_2
        tmp_hanna_point_list = []
        if pos_2_list[-1] != max_pos_2:
            pos_2_list.append(max_pos_2)
        for j,pos_2 in enumerate(pos_2_list):
            pre_pos_x,pre_pos_y = pos_1,pre_pos_2
            pos_x,pos_y = pos_1,pos_2
            flag = 0
            if pos_2 == min_pos_2:
                if (pos_x,pos_y) in pin_dict:
                    tmp_hanna_point_list.append([pos_x,pos_y,pos_x,pos_y,pin_dict[(pos_x,pos_y)]])
                continue
            hanna_point_info[id_num_hanna_points] = [pre_pos_x,pre_pos_y,pos_x,pos_y]#flag 0:v  1:h
            #########create group hanna point###########
            #########connect to pin in this line###########
            flag_1,flag_2 = pre_pos_y,pos_y
            pre_pin_pos = (pre_pos_x,pre_pos_y)
            if pre_pin_pos in pin_dict:
                pre_pin = pin_dict[pre_pin_pos]
                route_edge_us.append(pre_pin)
                route_edge_vs.append(id_num_hanna_points)
                route_edge_us.append(id_num_hanna_points)
                route_edge_vs.append(pre_pin)
                flag_1 = id2pinpos_y[pinpos_y2id[pre_pos_y]+1]
            if (pos_x,pos_y) in pin_dict:
                pin = pin_dict[(pos_x,pos_y)]
                route_edge_us.append(pin)
                route_edge_vs.append(id_num_hanna_points)
                route_edge_us.append(id_num_hanna_points)
                route_edge_vs.append(pin)
                flag_2 = id2pinpos_y[pinpos_y2id[pos_y]-1]
            tmp_hanna_point_list.append([pre_pos_x,flag_1,pos_x,flag_2,id_num_hanna_points])
            if (pos_x,pos_y) in pin_dict:
                tmp_hanna_point_list.append([pos_x,pos_y,pos_x,pos_y,pin_dict[(pos_x,pos_y)]])
            # if j == len(pos_2_list) and (not pos_2 == max_pos_2):

            id_num_hanna_points+=1
            pre_pos_2 = pos_2
            #########connect to pin in this line###########
        #########connect to pin in pre line###########
        tmp_id = 0
        tmp_pre_id = 0
        while tmp_pre_id < len(tmp_pre_hanna_point_list) and tmp_id < len(tmp_hanna_point_list):
            pre_pos_x_1,pre_pos_y_1,pos_x_1,pos_y_1,id_1 = tmp_pre_hanna_point_list[tmp_pre_id]
            pre_pos_x_2,pre_pos_y_2,pos_x_2,pos_y_2,id_2 = tmp_hanna_point_list[tmp_id]
            if intersect(pre_pos_x_1,pre_pos_y_1,pos_x_1,pos_y_1,pre_pos_x_2,pre_pos_y_2,pos_x_2,pos_y_2):
                route_edge_us.append(id_1)
                route_edge_vs.append(id_2)
                route_edge_us.append(id_2)
                route_edge_vs.append(id_1)
            if pre_pos_x_1 == pos_x_1:
                if pos_y_1 < pos_y_2:
                    tmp_pre_id+=1
                else:
                    tmp_id+=1
            else:
                if pos_x_1 < pos_x_2:
                    tmp_pre_id+=1
                else:
                    tmp_id+=1
        tmp_pre_hanna_point_list = tmp_hanna_point_list.copy()
                
    # print(hanna_point_info)
    # print(route_edge_us)
    # print(route_edge_vs)
    return route_edge_us,route_edge_vs,id_num_hanna_points
        #########connect to pin in pre line###########




def build_hanan_grid(pin_xs,pin_ys,nodes,num_hanna_points):
    pre_num_hanna_points = num_hanna_points
    pos_x_dict = {}
    pos_y_dict = {}
    pin_point_dict = {}
    hanna_point_info = {}
    pin_dict = {}
    route_edge_us = []
    route_edge_vs = []
    pin_edge_nodes = []
    pin_edge_hanna_points = []
    min_pos_x = min(pin_xs)
    min_pos_y = min(pin_ys)
    max_pos_x = max(pin_xs)
    max_pos_y = max(pin_ys)
    for pin_x,pin_y,node in zip(pin_xs,pin_ys,nodes):
        pos_x_dict.setdefault(pin_x,[]).append(pin_y)
        pos_y_dict.setdefault(pin_y,[]).append(pin_x)
        if (pin_x,pin_y) in pin_dict:
            pin_edge_hanna_points.append(pin_dict[(pin_x,pin_y)])
        else:
            pin_dict[(pin_x,pin_y)] = num_hanna_points
            pin_edge_hanna_points.append(num_hanna_points)
            num_hanna_points+=1
        pin_point_dict[(pin_x,pin_y)] = node
        pin_edge_nodes.append(node)
    
    route_edge_us,route_edge_vs,num_hanna_points = build_group_hanna_point(pos_x_dict,hanna_point_info,max_pos_y,min_pos_y,num_hanna_points,'x',
                            route_edge_us,route_edge_vs,pin_dict)


    return route_edge_us,route_edge_vs,pin_edge_nodes,pin_edge_hanna_points,num_hanna_points - pre_num_hanna_points

def transform_graph2edges(graph):
    num_nets = graph.num_nodes(ntype='net')
    nets,cells = graph.edges(etype='pinned')
    edges_feats = graph.edges['pinned'].data['feats']
    edges = {}
    # iter_info = tqdm.tqdm(zip(nets,cells,edges_feats),total=len(nets))
    iter_info = zip(nets,cells,edges_feats)
    for net,cell,pin_feats in iter_info:
        pin_px, pin_py, pin_io = pin_feats
        edges.setdefault(net.item(),[]).append([cell.item(),pin_px.item(), pin_py.item(), pin_io.item()])
    return edges

def build_route_graph(graph,node_pos):
    # edges = get_edge()
    # node_pos = get_pos()
    edges = transform_graph2edges(graph)
    # edge_iter = tqdm.tqdm(edges.items(),total=len(edges))
    edge_iter = edges.items()
    us = []
    vs = []
    route_edge_us = []
    route_edge_vs = []
    pin_edge_nodes = []
    pin_edge_hanna_points = []
    num_hanna_points = 0
    # total_time = 0
    # total_init_time = 0
    # total_route_time = 0
    # total_append_time = 0
    for net,list_node_feats in edge_iter:
        # total_a = time.time()
        pin_xs = []
        pin_ys = []
        nodes = []
        for node,pin_px,pin_py,pin_io in list_node_feats:
            us.append(node)
            vs.append(net)
            nodes.append(node)
            px,py = node_pos[node,:]
            pin_xs.append(px + pin_px)
            pin_ys.append(py + pin_py)
        # total_init_time += time.time() - total_a
        # total_b = time.time()
        sub_route_edge_us,sub_route_edge_vs,sub_pin_edge_nodes,sub_pin_edge_hanna_points,sub_num_hanna_point = build_hanan_grid(pin_xs,pin_ys,nodes,num_hanna_points)
        # total_route_time += time.time() - total_b
        # total_b = time.time()
        route_edge_us.extend(sub_route_edge_us)
        route_edge_vs.extend(sub_route_edge_vs)
        pin_edge_nodes.extend(sub_pin_edge_nodes)
        pin_edge_hanna_points.extend(sub_pin_edge_hanna_points)
        num_hanna_points+=sub_num_hanna_point
        # total_append_time += time.time() - total_b
        # total_time += time.time() - total_a
    graph = dgl.heterograph({
        ('cell','pins','net'):(us,vs),
        ('net','pinned','cell'):(vs,us),
        ('cell','point-to','hanna'):(pin_edge_nodes,pin_edge_hanna_points),
        ('hanna','point-from','cell'):(pin_edge_hanna_points,pin_edge_nodes),
        ('hanna','connect','hanna'):(route_edge_us,route_edge_vs)
    })
    # print(f"total init time {total_init_time}s averange time is {total_init_time/len(edges)}s")
    # print(f"total route time {total_route_time}s averange time is {total_route_time/len(edges)}s")
    # print(f"total append time {total_append_time}s averange time is {total_append_time/len(edges)}s")
    # print(f"total time {total_time}s averange time is {total_time/len(edges)}s")
    return graph

def load_graph(netlist_name):
    edges = get_edge(netlist_name)
    node_pos = get_pos(netlist_name)
    edge_iter = tqdm.tqdm(edges.items(),total=len(edges))
    us = []
    vs = []
    pins_feats = []
    net_degree = []
    for net,list_node_feats in edge_iter:
        net_degree.append(len(list_node_feats))
        for node, pin_px, pin_py, pin_io in list_node_feats:
            us.append(node)
            vs.append(net)
            pins_feats.append([pin_px, pin_py, pin_io])
    pins_feats = torch.tensor(pins_feats,dtype=torch.float32)
    node_pos = torch.tensor(node_pos,dtype=torch.float32)
    hetero_graph = dgl.heterograph({
        ('cell','pins','net'):(us,vs),
        ('net','pinned','cell'):(vs,us),
    },num_nodes_dict={'cell':len(node_pos),'net':len(net_degree)})
    hetero_graph.nodes['cell'].data['pos'] = node_pos
    hetero_graph.edges['pinned'].data['feats'] = pins_feats
    partition_list = partition_graph(hetero_graph)
    # route_graph = build_route_graph(hetero_graph,node_pos)
    # return route_graph

    list_hetero_graph = []
    list_route_graph = []
    iter_partition_list = tqdm.tqdm(partition_list, total=len(partition_list))
    total_route_time = 0
    total_sub_time = 0
    total_init_time = 0
    total_time = 0
    num_hanna_point = 0
    num_hanna_edges = 0
    for partition in iter_partition_list:
        # total_a = time.time()
        partition_set = set(partition)
        new_net_degree_dict = {}
        for net_id, node_id in zip(*[ns.tolist() for ns in hetero_graph.edges(etype='pinned')]):
            if node_id in partition_set:
                new_net_degree_dict.setdefault(net_id, 0)
                new_net_degree_dict[net_id] += 1
        keep_nets_id = np.array(list(new_net_degree_dict.keys()))
        keep_nets_degree = np.array(list(new_net_degree_dict.values()))
        # good_nets = np.abs(net_degree[keep_nets_id] - keep_nets_degree) < 1e-5
        # keep_nets_id = keep_nets_id[good_nets]
        # total_init_time += time.time() - total_a
        # total_b = time.time()
        part_hetero_graph = dgl.node_subgraph(hetero_graph, nodes={'cell': partition, 'net': keep_nets_id})
        # total_sub_time += time.time() - total_b
        list_hetero_graph.append(part_hetero_graph)
        sub_node_pos = part_hetero_graph.nodes['cell'].data['pos']
        # total_b = time.time()
        sub_route_graph = build_route_graph(part_hetero_graph,sub_node_pos)
        # total_route_time += time.time() - total_b
        list_route_graph.append(sub_route_graph)
        # total_time += time.time() - total_a
        # print(sub_route_graph)
        num_hanna_point += sub_route_graph.num_nodes(ntype='hanna')
        num_hanna_edges += sub_route_graph.num_edges(etype='connect')
    # print(f"total init time {total_init_time}s average time {total_init_time/len(partition_list)}s")
    # print(f"total sub time {total_sub_time}s average time {total_sub_time/len(partition_list)}s")
    # print(f"total route time {total_route_time}s average time {total_route_time/len(partition_list)}s")
    # print(f"total time {total_time}s average time {total_time/len(partition_list)}s")
    print(f"total hanna point {num_hanna_point}")
    print(f"total hanna edges {num_hanna_edges}")
    # return num_hanna_point,num_hanna_edges
    return list_hetero_graph,list_route_graph


if __name__ == '__main__':
    netlist_name_list = [
        'superblue1',
        # 'superblue2',
        # 'superblue3',
        # 'superblue5',
        # 'superblue6',
        # 'superblue7',
        # 'superblue9',
        # 'superblue11',
        # 'superblue14',
        # 'superblue16',
        # 'superblue19',
    ]
    netlist_info = []
    for netlist_name in netlist_name_list:
        # num_hanna_point,num_hanna_edges = load_graph(netlist_name)
        netlist_info.append([netlist_name,0,0])
    netlist_info = np.array(netlist_info)
    df = pd.DataFrame()
    df['netlist_name'] = netlist_info[:,0]
    df['num hanan points'] = netlist_info[:,1]
    df['hanan edges'] = netlist_info[:,2]
    df.to_excel("./group_hanna_info.xlsx",index=False)
    # print(graph)
    # with open("/root/test.pickle",'wb+') as fp:
    #     pickle.dump(graph,fp)
    # get_w()



def build_hanan_grid_simple(pin_xs,pin_ys,nodes,num_hanna_points):
    pin_point_dict = {}
    hanna_point_dict = {}
    for pin_x,pin_y,node in zip(pin_xs,pin_ys,nodes):
        pin_point_dict[(pin_x,pin_y)] = node
    sorted_pin_xs = pin_xs.copy()
    sorted_pin_ys = pin_ys.copy()
    sorted_pin_xs.sort()
    sorted_pin_ys.sort()
    id_hanna_point = 0
    for x,y in itertools.product(sorted_pin_xs,sorted_pin_ys):#这步必须保证是按序进行笛卡尔积的！！！！！
        if (x,y) in pin_point_dict:
            pin2node = pin_point_dict[(x,y)]
        else:
            pin2node = -1
        hanna_point_dict[(x,y)] = (id_hanna_point,pin2node)
        id_hanna_point+=1
    
    route_edge_us = []
    route_edge_vs = []
    pin_edge_nodes = []
    pin_edge_hanna_points = []
    num_pins = len(pin_xs)
    for id_hanna_point,pin2node in hanna_point_dict.values():
        id_x = id_hanna_point % num_pins
        id_y = int(id_hanna_point / num_pins)
        for dx,dy in [[0,1],[0,-1],[1,0],[-1,0]]:
            id_x_ = id_x + dx
            id_y_ = id_y + dy
            if id_x_ < 0 or id_x_ >= num_pins or id_y_ < 0 or id_y_ >= num_pins:
                continue
            id_hanna_point_ = id_x_ + id_y_ * num_pins
            route_edge_us.append(id_hanna_point + num_hanna_points)
            route_edge_vs.append(id_hanna_point_ + num_hanna_points)
        if pin2node != -1:
            pin_edge_nodes.append(pin2node)
            pin_edge_hanna_points.append(id_hanna_point + num_hanna_points)
    return route_edge_us,route_edge_vs,pin_edge_nodes,pin_edge_hanna_points,id_hanna_point