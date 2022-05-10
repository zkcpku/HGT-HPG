import re
import numpy as np
import pickle
from tqdm import tqdm
import string
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from java_parser.myparser import heterograph_graph_parser, heterograph_graph_parser_subtoken
from python_parser.json_utils import serialize_graph_dicts, deserialize_graph_dicts


# step2
from utils.vocab import Vocab, VocabEntry
import torch
import dgl

def tokenizer(name):
    def camel_case_split(identifier):
        matches = re.finditer(
            '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
            identifier,
        )
        return [m.group(0) for m in matches]
    blocks = []
    for underscore_block in re.split(' |_', name):
        blocks.extend(camel_case_split(underscore_block))
    return blocks

def step1_main():
    RAW_PATH = '/home/username/workspace/HGT-DGL/data/java-small/raw/'
    STEP1_PATH = '/home/username/workspace/HGT-DGL/data/java-small/step1/'
    classes = ['train', 'valid', 'test']
    BATCH_SIZE = 150000

    for cls in classes:
        this_PATH = os.path.join(RAW_PATH, cls)
        all_files = [os.path.join(this_PATH, f)
                        for f in os.listdir(this_PATH) if 'json' in f]
        data = []
        for data_path in all_files:
            with open(data_path,'r',encoding='gb18030') as f:
                data += json.load(f)
        len_data = len(data)
        print('{} has {} files'.format(cls, len(all_files)))
        print('{} has {} raw data'.format(cls, len_data))

        print('Processing {}...'.format(cls))
        
        
        save_data = []
        start_id = 0
        error_num = 0
        pbar = tqdm(total=len_data,ncols=60)
        for i in range(len_data):
            # import pdb;pdb.set_trace()
            if (i+1) % BATCH_SIZE == 0:
                with open(os.path.join(STEP1_PATH, cls + '_' + str(start_id) + 'to' + str(i) + '.json'), 'w') as f:
                    json.dump(save_data, f)
                save_data = []
                start_id = i+1
            pbar.update(1)
            # import pdb; pdb.set_trace()
            try:
                code_src = data[i]['code']
                all_graph_dict = heterograph_graph_parser_subtoken(
                    code_src, False, share_subtoken=False)
                for each_node_i in range(len(all_graph_dict[0]['all_node'])):
                    if all_graph_dict[0]['all_node'][each_node_i]['type'] == 'subtoken':
                        all_graph_dict[0]['all_node'][each_node_i]['name'] = all_graph_dict[0]['all_node'][each_node_i]['name'].lower()
                    

                data[i]['graph_dict'] = serialize_graph_dicts(
                    all_graph_dict)
                if type(data[i]['name']) != list:
                    this_docstring_tokens = [data[i]['name']]
                else:
                    this_docstring_tokens = data[i]['name']
                this_docstring_tokens = [tokenizer(e) for e in this_docstring_tokens]
                this_docstring_tokens = [e for each_tokens in this_docstring_tokens for e in each_tokens]
                data[i]['name'] = [
                    e.lower() for e in this_docstring_tokens]

            
            except Exception as e:
                data[i]['err_msg'] = str(e) + ' ' + str(type(e))
                error_num += 1
                pbar.set_description('Error: {}'.format(error_num))
            save_data.append(data[i])

        with open(os.path.join(STEP1_PATH, cls + '_' + str(start_id) + 'to' + str(len(data)) + '.json'), 'w') as f:
            json.dump(save_data, f)
        save_data = []


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

def step2_main():
    # save_zero_dict and vocab and url
    STEP1_PATH = '/home/username/workspace/HGT-DGL/data/java-small/step1/'
    STEP2_PATH = '/home/username/workspace/HGT-DGL/data/java-small/step2/'
    all_files = [os.path.join(STEP1_PATH, f)
                 for f in os.listdir(STEP1_PATH) if 'json' in f and 'step2' not in f]
    
    train_name = []
    train_type = []
    train_field = []
    train_tgt_sents = []
    
    all_edge_key_in_each_graphs = []
    for each_file in all_files:
        print('Processing {}...'.format(each_file))
        with open(each_file, 'r') as f:
            data = json.load(f)
        if 'train' in each_file:
            for each_data in tqdm(data):
                if 'err_msg' in each_data:
                    continue
                each_graph_dict = deserialize_graph_dicts(each_data['graph_dict'])
                train_name.append([n['name'] for n in each_graph_dict[0]['all_node']])
                train_type.append([n['type'] for n in each_graph_dict[0]['all_node']])
                train_field.append([n['toParentField'] for n in each_graph_dict[0]['all_node']])
                train_tgt_sents.append(each_data['name'])
                all_edge_key_in_each_graphs.append(list(each_graph_dict[1].keys()))
        else:
            for each_data in tqdm(data):
                if 'err_msg' in each_data:
                    continue
                each_graph_dict = deserialize_graph_dicts(
                    each_data['graph_dict'])
                all_edge_key_in_each_graphs.append(
                    list(each_graph_dict[1].keys()))


    # save_pre_dict
    with open(STEP2_PATH + 'pre_vocab','w') as f:
        json.dump({'train_name': train_name, 'train_type': train_type, 'train_field': train_field, 'train_tgt_sents': train_tgt_sents}, f)

    # save_zero_dict
    zero_dict = generate_zero_dict(all_edge_key_in_each_graphs)
    with open(STEP2_PATH + 'zero_dict', 'wb') as f:
        pickle.dump(zero_dict, f)
    
    # save_vocab
    src_vocab = VocabEntry.from_corpus(train_name, size=150000)
    tgt_vocab = VocabEntry.from_corpus(train_tgt_sents, size=150000)
    # with open(STEP2_PATH + 'vocab.pkl', 'wb') as f:
    #     pickle.dump({'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}, f)
    with open(STEP2_PATH + 'src_vocab.pkl', 'wb') as f:
        pickle.dump(src_vocab, f)
    with open(STEP2_PATH + 'tgt_vocab.pkl', 'wb') as f:
        pickle.dump(tgt_vocab, f)

    # save_node_edge_dict
    node_edge_dict = zero_dict2node_edge_dict(zero_dict)
    # with open(STEP2_PATH + 'node_edge_dict.pkl', 'wb') as f:
    #     pickle.dump(node_edge_dict, f)
    with open(STEP2_PATH + 'node_edge_dict.json','w') as f:
        json.dump(node_edge_dict, f)



def step3_main():
    # fliter only graph_dict and docstring_tokens
    STEP1_PATH = '/home/username/workspace/HGT-DGL/data/java-small/step1/'
    STEP2_PATH = '/home/username/workspace/HGT-DGL/data/java-small/step2/'
    STEP3_PATH = '/home/username/workspace/HGT-DGL/data/java-small/step3/'
    all_files = [os.path.join(STEP1_PATH, f)
                 for f in os.listdir(STEP1_PATH) if 'json' in f and 'step2' not in f and 'step3' not in f]
    for each_file in all_files:
        print('Processing {}...'.format(each_file))
        with open(each_file, 'r') as f:
            data = json.load(f)
        save_data = []
        pbar = tqdm(total=len(data))
        for i, each_data in enumerate(data):
            pbar.update(1)
            if 'err_msg' in each_data:
                continue
            each_graph_dict = each_data['graph_dict']
            save_data.append({
                'graph_dict': each_graph_dict,
                'tgt': each_data['name']
            })
        
        with open(os.path.join(STEP3_PATH, each_file.split('/')[-1]), 'w') as f:
            json.dump(save_data, f)

if __name__ == '__main__':
    step1_main()
    step2_main()
    step3_main()

        
