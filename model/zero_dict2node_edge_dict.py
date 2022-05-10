import pickle
import json

import dgl

zero_dict = pickle.load(open('/home/zhangkechi/workspace/HGT-DGL/data/ogbg_code/graph2seq/hete_subtoken/zero_dict.pkl', 'rb'))
len(zero_dict)
zero_graph = dgl.heterograph(zero_dict)
G_node_dict, G_edge_dict = {}, {}
for ntype in list(set(zero_graph.ntypes)):
    G_node_dict[ntype] = len(G_node_dict)
for etype in list(set(zero_graph.etypes)):
    G_edge_dict[etype] = len(G_edge_dict)

with open('/home/zhangkechi/workspace/HGT-DGL/data/ogbg_code/graph2seq/hete_subtoken/node_edge_dict.json','w') as f:
    json.dump({'node': G_node_dict, 'edge': G_edge_dict}, f)