import json
import numpy as np
import torch
import dgl


def serialize_graph_dicts(parser_output):
    # [nodes, new_dict, each_node_dict]
    parser_output = list(parser_output)
    parser_output[1] = [(k, (v[0].tolist(), v[1].tolist()))
                        for k, v in parser_output[1].items()]
    return json.dumps(parser_output)


def deserialize_graph_dicts(serialized_graph_dicts):
    # [nodes, new_dict, each_node_dict]
    serialized_graph_dicts = json.loads(serialized_graph_dicts)

    serialized_graph_dicts[1] = {tuple(k): (
        np.array(v[0]), np.array(v[1])) for k, v in serialized_graph_dicts[1]}
    return serialized_graph_dicts


def dict2graph(new_dict, each_node_dict, zero_dict=None):
    for k in zero_dict:
        if k not in new_dict:
            new_dict[k] = zero_dict[k]
    out_g = dgl.heterograph(new_dict)
    for ntype in out_g.ntypes:
        out_g.nodes[ntype].data['idx'] = torch.tensor(
            each_node_dict[ntype] if ntype in each_node_dict else [])
    return out_g
