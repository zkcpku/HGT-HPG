# https://discuss.pytorch.org/t/dataloader-for-a-folder-with-multiple-files-pytorch-solutions-that-is-equivalent-to-tfrecorddataset-in-tf2-0/70512
# https://stackoverflow.com/questions/53477861/pytorch-dataloader-multiple-data-source
# https://stackoverflow.com/questions/60127632/load-multiple-npy-files-size-10gb-in-pytorch
# https://www.zhihu.com/question/356829360/answer/902681727
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.data as data

from python_parser.json_utils import deserialize_graph_dicts, dict2graph

import dgl

from utils.vocab import Vocab, VocabEntry
from utils.nn_utils import word2id


class SingeJsonDataset(data.Dataset):
    # implement a single json dataset here...
    def __init__(self,json_file) -> None:
        super().__init__()
        self.json_file = json_file
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)
            # [label, graph_json, code_src, file_path]
        self.data = [[e[0], deserialize_graph_dicts(e[1])] for e in self.data]
    
    def __getitem__(self, index):
        origin_data = self.data[index]
        graph_dict = origin_data[1]
        target_label = origin_data[0]
        edge_dict = graph_dict[1]
        each_node_dict = graph_dict[2]
        node_info = graph_dict[0]
        return (node_info, edge_dict, each_node_dict, target_label)

    def __len__(self):
        return len(self.data)


def get_dataset(paths):
    list_of_datasets = []
    for j in paths:
        if not j.endswith('.json'):
            continue  # skip non-json files
        list_of_datasets.append(SingeJsonDataset(
            json_file=j))
    # once all single json datasets are created you can concat them into a single one:
    multiple_json_dataset = data.ConcatDataset(list_of_datasets)
    return multiple_json_dataset



def get_collate_fn(src_vocab, zero_dict, count_node_nums_type = ['subtoken'], device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    if type(count_node_nums_type) != list:
        count_node_nums_type = [count_node_nums_type]
    def collate_fn(data):
        # len(data) == batch_size
        node_info, edge_dict, each_node_dict, target_label = zip(*data)
        bs = len(node_info)
        input_graphs = []
        # all_subtoken_nums = []
        all_node_nums = []
        for i in range(bs):
            g = dict2graph(edge_dict[i], each_node_dict[i], zero_dict)
            all_node_info = node_info[i]['all_node']
            for ntype in g.ntypes:
                this_idx_list = g.nodes[ntype].data['idx'].tolist()
                if len(this_idx_list) == 0:
                    g.nodes[ntype].data['name'] = torch.tensor([])
                    continue
                this_info = [all_node_info[i] for i in this_idx_list]

                this_name_info = [e['name'] for e in this_info]
                this_name_vars = word2id(this_name_info, src_vocab)

                g.nodes[ntype].data['name'] = torch.tensor(this_name_vars)
            
            all_node_nums.append({k: g.num_nodes(k) for k in count_node_nums_type})
            
            # all_identifier_nums.append(g.num_nodes('identifier'))
            # all_subtoken_nums.append(g.num_nodes(count_node_nums_type))
            input_graphs.append(g)
        
        batch_graph = dgl.batch(input_graphs)
        # batch_graph = batch_graph.to(device)

        target_label = torch.tensor(target_label, dtype=torch.long)
        return batch_graph, target_label, all_node_nums
    return collate_fn

def unit_test():
    import psutil
    import time
    zero_dict_path = '/home/username/workspace/HGT-DGL/data/codenet/python/step3/zero_dict.pkl'
    zero_dict = pickle.load(open(zero_dict_path, 'rb'))
    src_vocab_path = '/home/username/workspace/HGT-DGL/data/codenet/python/step3/vocab.pkl'
    src_vocab = pickle.load(open(src_vocab_path, 'rb'))
    paths = [
        '/home/username/workspace/HGT-DGL/data/codenet/python/step2/devset.json']
    t1 = time.time()
    dataset = get_dataset(paths)
    t2 = time.time()
    def memory_usage():
        mem_available = psutil.virtual_memory().available
        mem_process = psutil.Process(os.getpid()).memory_info().rss
        return round(mem_process / 1024 / 1024, 2), round(mem_available / 1024 / 1024, 2)
    print('time cost:', t2 - t1)
    print('memory usage:', memory_usage())
    # import pdb;pdb.set_trace()
    

    # https://medium.com/geekculture/pytorch-datasets-dataloader-samplers-and-the-collat-fn-bbfc7c527cf1
    # def collate_fn(data):
    #     # data is a list of tuple (node_info, input_graph, target_sent)
    #     # node_info is a list of dicts
    #     # input_graph is a dgl graph
    #     # target_sent is a list of ints
    #     # import pdb;pdb.set_trace()
    #     node_info, input_graph, target_sent = zip(*data)
    #     all_identifier_nums = [g.num_nodes('identifier') for g in input_graph]
    #     input_graph = dgl.batch(input_graph)
    #     return node_info, input_graph, target_sent, all_identifier_nums
    collate_fn = get_collate_fn(src_vocab, zero_dict)
    # , num_workers = 4, prefetch_factor = 10
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=4, prefetch_factor=10, pin_memory=True)
    
    return dataloader
    # import pdb;pdb.set_trace()
    for batch_idx, sample in enumerate(dataloader):
        batch_graph, target_sent, all_node_nums = sample
        # print(batch_idx, sample)
        print(memory_usage())
        import pdb;pdb.set_trace()

        break


if __name__ == "__main__":
    unit_test()
