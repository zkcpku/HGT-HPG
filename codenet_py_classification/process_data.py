import re
import numpy as np
import pickle
from tqdm import tqdm
import string
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_parser.myparser import heterograph_graph_parser, heterograph_graph_parser_subtoken
from python_parser.json_utils import serialize_graph_dicts, deserialize_graph_dicts


# step2
from utils.vocab import Vocab, VocabEntry
import torch
import dgl

# split_data and get_data_content
def main_step1():
    ROOT_PATH = '/home/username/workspace/data/Project_CodeNet_Python800/'
    STEP1_PATH = '/home/username/workspace/HGT-DGL/data/codenet/python/no_share_subtoken/step1'

    cls_dict = {}
    # cls, content, file_path
    all_data = []
    for cls in tqdm(os.listdir(ROOT_PATH)):
        if cls == '.DS_Store':
            continue
        
        if cls not in cls_dict:
            cls_dict[cls] = len(cls_dict)
        
        cls_path = os.path.join(ROOT_PATH, cls)
        for file in os.listdir(cls_path):
            file_path = os.path.join(cls_path, file)
            with open(file_path, 'r') as f:
                content = f.read()
            all_data.append((cls_dict[cls], content, file_path))
    
    with open(os.path.join(STEP1_PATH, 'cls_dict.json'), 'w') as f:
        json.dump(cls_dict, f)
    
    with open(os.path.join(STEP1_PATH, 'all_data.json'), 'w') as f:
        json.dump(all_data, f)

    trainset, devset, testset = [[], [], []]
    for i, (cls, content, file_path) in enumerate(all_data):
        if i % 5 == 3:
            devset.append((cls, content, file_path))
        elif i % 5 == 4:
            testset.append((cls, content, file_path))
        else:
            trainset.append((cls, content, file_path))
    
    with open(os.path.join(STEP1_PATH, 'trainset.json'), 'w') as f:
        json.dump(trainset, f)
    
    with open(os.path.join(STEP1_PATH, 'devset.json'), 'w') as f:
        json.dump(devset, f)
    
    with open(os.path.join(STEP1_PATH, 'testset.json'), 'w') as f:
        json.dump(testset, f)
            

def serialize_graph_dicts(parser_output):
    # [nodes, new_dict, each_node_dict]
    parser_output = list(parser_output)
    parser_output[1] = [(k, (v[0].tolist(), v[1].tolist()))
                    for k, v in parser_output[1].items()]
    return json.dumps(parser_output)

def deserialize_graph_dicts(serialized_graph_dicts):
    # [nodes, new_dict, each_node_dict]
    serialized_graph_dicts = json.loads(serialized_graph_dicts)

    serialized_graph_dicts[1] = {tuple(k): (np.array(v[0]), np.array(v[1])) for k,v in serialized_graph_dicts[1]}
    return serialized_graph_dicts

# graph_json
# [label, code_src, file_path]
def main_step2():
    STEP1_PATH = '/home/username/workspace/HGT-DGL/data/codenet/python/no_share_subtoken/step1'
    STEP2_PATH = '/home/username/workspace/HGT-DGL/data/codenet/python/no_share_subtoken/step2'
    cls_dataset = ['trainset','devset','testset']
    error_msg = []
    for cls_dataset_name in cls_dataset:
        dataset_path1 = os.path.join(STEP1_PATH, cls_dataset_name + '.json')
        dataset1 = json.load(open(dataset_path1, 'r'))
        dataset2 = []
        pbar = tqdm(dataset1)
        for each_data in pbar:
            label, code_src, file_path = each_data
            try:
                graph_json = heterograph_graph_parser_subtoken(
                    code_src, False, share_subtoken=False)
                graph_json = serialize_graph_dicts(graph_json)
                dataset2.append([label, graph_json, code_src, file_path])
            except Exception as e:
                error_msg.append((str(e), file_path, cls_dataset_name))
                pbar.set_description('Error: {}, num: {}'.format(str(e), len(error_msg)))

                continue

        with open(os.path.join(STEP2_PATH, cls_dataset_name + '.json'), 'w') as f:
            json.dump(dataset2, f)
    with open(os.path.join(STEP2_PATH, 'error_msg.json'), 'w') as f:
        json.dump(error_msg, f)
        # import pdb; pdb.set_trace()


def generate_zero_dict(all_edge_key_in_each_graphs):
    g_keys_set = [
        e for each_sample in all_edge_key_in_each_graphs for e in each_sample]

    g_keys_set = list(set(g_keys_set))
    zero_dict = {k: (np.array([]), np.array([]))
                 for k in g_keys_set}
    # zero_dict
    return zero_dict


def zero_dict2node_edge_dict(zero_dict):
    zero_graph = dgl.heterograph(zero_dict)
    G_node_dict, G_edge_dict = {}, {}
    for ntype in list(set(zero_graph.ntypes)):
        G_node_dict[ntype] = len(G_node_dict)
    for etype in list(set(zero_graph.etypes)):
        G_edge_dict[etype] = len(G_edge_dict)

    return {'node': G_node_dict, 'edge': G_edge_dict}

def main_step3():
    STEP2_PATH = '/home/username/workspace/HGT-DGL/data/codenet/python/no_share_subtoken/step2/'
    STEP3_PATH = '/home/username/workspace/HGT-DGL/data/codenet/python/no_share_subtoken/step3/'

    cls_dataset = ['trainset','devset','testset']

    # pre_vocab
    train_name = []
    train_dataset2 = json.load(open(os.path.join(STEP2_PATH, 'trainset.json'), 'r'))
    train_name = [deserialize_graph_dicts(e[1])[0]['all_node'] for e in train_dataset2]
    train_name = [[n['name'] for n in graph_nodes] for graph_nodes in train_name]
    # train_name = [n['name'] for e in train_dataset2 for n in deserialize_graph_dicts(e[1])[0]['all_node']]
    with open(STEP3_PATH + 'pre_vocab.json', 'w') as f:
        json.dump({'train_name': train_name}, f)

    # zero_dict
    all_edge_key_in_each_graphs = []
    for cls_dataset_name in cls_dataset:
        dataset_path2 = os.path.join(STEP2_PATH, cls_dataset_name + '.json')
        dataset2 = json.load(open(dataset_path2, 'r'))
        print(cls_dataset_name, '\t', len(dataset2))
        dataset2 = [deserialize_graph_dicts(e[1]) for e in dataset2]
        this_edge_key_in_each_graphs = [
            list(each_graph_dict[1].keys()) for each_graph_dict in dataset2]
        this_edge_key_in_each_graphs = list(
            set([e for each_sample in this_edge_key_in_each_graphs for e in each_sample]))
        all_edge_key_in_each_graphs.append(this_edge_key_in_each_graphs)
    zero_dict = generate_zero_dict(all_edge_key_in_each_graphs)
    with open(STEP3_PATH + 'zero_dict.pkl', 'wb') as f:
        pickle.dump(zero_dict, f)

    
    # vocab
    all_names = [e for each_sample in train_name for e in each_sample]
    print("origin vocab size:", len(set(all_names)))
    src_vocab = VocabEntry.from_corpus(train_name, size=150000)
    with open(STEP3_PATH + 'vocab.pkl', 'wb') as f:
        pickle.dump(src_vocab, f)


    # node_edge_dict
    node_edge_dict = zero_dict2node_edge_dict(zero_dict)
    with open(STEP3_PATH + 'node_edge_dict.json', 'w') as f:
        json.dump(node_edge_dict, f)



if __name__ == '__main__':
    main_step1()
    main_step2()
    main_step3()
