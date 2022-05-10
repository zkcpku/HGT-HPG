import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
# from torch.nn.parallel import DistributedDataParallel as DataParallel
import torch.optim as optim
import dgl
from dgl.data.utils import load_graphs

import time
import random
from tqdm import tqdm
import pickle
import json

from config import *
import utils.nn_utils as nn_utils
from utils.nn_utils import batch_iter
from utils.vocab import VocabEntry, Vocab
from utils.log_utils import logWriter

from utils.eval_utils import *
# from dgl.data import DGLDataset
# from dgl import save_graphs, load_graphs
# from dgl.data.utils import makedirs, save_info, load_info
from transformers import AdamW, get_linear_schedule_with_warmup

from model.graph2seq import HGTCopyTransformer, HGTCopyTransformer_only_subtoken


# local_rank = int(os.environ["LOCAL_RANK"])
# mylogger = logWriter(my_config.save['log_path'] + '_' + str(local_rank))
mylogger = logWriter(my_config.save['log_path'])

# mylogger.write_now_time()
mylogger.write(my_config)

def set_seed(seed=36):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_multi_iter(gs_paths, info_paths, model, optimizer, epoch):
    assert len(gs_paths) == len(info_paths)
    for i in tqdm(range(len(gs_paths))):
        t1 = time.time()
        gs_path = gs_paths[i]
        info_path = info_paths[i]
        gs, labels = load_graphs(gs_path)
        infos = json.load(open(info_path))
        infos = [e[0] for e in infos]

        train_data = list(zip(gs, infos))
        del gs, labels, infos
        t2 = time.time()
        mylogger.write("load graph {} cost {}".format(gs_path, t2-t1))
        mylogger.write("load info {}".format(info_path))
        t1 = time.time()
        train_iter(train_data, model, optimizer, epoch)
        #torch.cuda.empty_cache()
        t2 = time.time()
        mylogger.write("train {} cost {}".format(gs_path, t2-t1))
        if my_config.save_model:
            torch.save(model.state_dict(),
                       my_config.save['save_model_path'] + '_' + str(epoch) +'_' + str(i))
        del train_data

def train_iter(dataset, model, optimizer, epoch):
    model.train()
    mylogger.write("start training...")
    bar = tqdm(total=len(dataset))
    running_loss = 0.0
    for batch_data in batch_iter(dataset, my_config.optimizer['bs'], True, sort=False):
        try:
            bar.update(my_config.optimizer['bs'])
            # import pdb; pdb.set_trace()
            # exit()
            # inputs, labels = batch_data
            inputs = [e[0] for e in batch_data]
            identifier_nums = [g.num_nodes('identifier' if not hasattr(my_config, 'split_node_type') else my_config.split_node_type) for g in inputs]
            inputs = dgl.batch(inputs)
            
            labels = [e[1] for e in batch_data]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            scroes = model(inputs, identifier_nums, labels)
            loss = -torch.sum(scroes)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(), my_config.optim['max_grad_norm'])

            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            this_time_loss = float(loss.item())
            running_loss += this_time_loss
            bar.set_description("epoch {} loss {}".format(epoch, this_time_loss))
            #torch.cuda.empty_cache()
        except Exception as e:
            mylogger.write(e)
            if my_config.save_model:
                torch.save(model.state_dict(),
                       my_config.save['save_model_path'] + '_' + str(epoch) + '_' + 'error')
            # mylogger.write(batch_data)
            #torch.cuda.empty_cache()
            continue
        # #torch.cuda.empty_cache()

    mylogger.write("epoch {} loss {}".format(epoch, running_loss / (len(dataset) // my_config.optimizer['bs'])))


def test_iter(gs_path, info_path, model, epoch, DEBUG= False, write_to_file = False):
    if isinstance(model, DataParallel):
        model = model.module
    t1 = time.time()
    gs, labels = load_graphs(gs_path)
    infos = json.load(open(info_path))
    infos = [e[0] for e in infos]

    dataset = list(zip(gs, infos))
    del gs, labels, infos
    t2 = time.time()
    mylogger.write("load graph {} cost {}".format(gs_path, t2-t1))
    mylogger.write("load info {}".format(info_path))
    
    model.eval()
    mylogger.write("start testing...")
    with torch.no_grad():
        bar = tqdm(total=len(dataset))
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_inputs = []
        for batch_data in batch_iter(dataset, my_config.optimizer['test_bs'], False, sort=False):
            bar.update(my_config.optimizer['test_bs'])
            inputs = [e[0] for e in batch_data]
            labels = [e[1] for e in batch_data]
            for i,e in enumerate(inputs):
                outs = model.sample(e, 
                                    max_len = my_config.sample['decode_max_time_step'], 
                                    sample_size = my_config.sample['sample_size'], 
                                    mode = my_config.sample['mode'])
                all_inputs.append(e)
                all_labels.append(labels[i])
                all_preds.append(outs)
            if DEBUG:
                break # only test one batch
        mylogger.write("Epoch " + str(epoch) + " test:")
        test_rst = corpus_test(all_labels, all_preds)
        mylogger.write(test_rst)
        mylogger.write("\n")
        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if write_to_file:
            with open(my_config.save['output_path'] + '_' + str(epoch) + str(now_time), 'w') as f:
                for i, e in enumerate(all_labels):
                    f.write(str(all_inputs[i]) + '\n')
                    f.write(str(e) + '\n')
                    f.write(str(all_preds[i]))
                    f.write('\n\n')

        mylogger.write('write to file : ' +
                       my_config.save['output_path'] + '_' + str(epoch) + str(now_time))

        if DEBUG:
            for i,e in enumerate(all_labels): # for debug
                mylogger.write(all_inputs[i])
                mylogger.write(e)
                mylogger.write(all_preds[i])

    return test_rst

def test_iter_batch(gs_path, info_path, model, epoch, DEBUG= False, write_to_file = False):
    if isinstance(model, DataParallel):
        model = model.module
    t1 = time.time()
    gs, labels = load_graphs(gs_path)
    infos = json.load(open(info_path))
    infos = [e[0] for e in infos]

    dataset = list(zip(gs, infos))
    del gs, labels, infos
    t2 = time.time()
    mylogger.write("load graph {} cost {}".format(gs_path, t2-t1))
    mylogger.write("load info {}".format(info_path))
    
    model.eval()
    mylogger.write("start testing...")
    with torch.no_grad():
        bar = tqdm(total=len(dataset))
        running_loss = 0.0
        all_labels = []
        all_preds = []
        # all_inputs = []
        for batch_data in batch_iter(dataset, my_config.optimizer['test_bs'], False, sort=False):
            bar.update(my_config.optimizer['test_bs'])
            inputs = [e[0] for e in batch_data]
            identifier_nums = [g.num_nodes('identifier' if not hasattr(my_config, 'split_node_type') else my_config.split_node_type) for g in inputs]
            # import pdb; pdb.set_trace()
            inputs = dgl.batch(inputs)
            labels = [e[1] for e in batch_data]
            outs_batch = model.sample_batch(inputs, identifier_nums,
                                            max_len = my_config.sample['decode_max_time_step'], 
                                            sample_size = my_config.sample['sample_size'], 
                                            mode = my_config.sample['mode'])
            # all_inputs.extend(e)
            all_labels.extend(labels)
            all_preds.extend(outs_batch)

            if DEBUG:
                break # only test one batch
        mylogger.write("Epoch " + str(epoch) + " test:")
        test_rst = corpus_test(all_labels, all_preds)
        mylogger.write(test_rst)
        mylogger.write("\n")
        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if write_to_file:
            with open(my_config.save['output_path'] + '_' + str(epoch) + str(now_time), 'w') as f:
                for i, e in enumerate(all_labels):
                    # f.write(str(all_inputs[i]) + '\n')
                    f.write(str(e) + '\n')
                    f.write(str(all_preds[i]))
                    f.write('\n\n')

        mylogger.write('write to file : ' +
                       my_config.save['output_path'] + '_' + str(epoch) + str(now_time))

        if DEBUG:
            for i,e in enumerate(all_labels): # for debug
                # mylogger.write(all_inputs[i])
                mylogger.write(e)
                mylogger.write(all_preds[i])

    return test_rst
        
def main(TRAIN = True):
    set_seed(my_config.seed)

    # load data
    # train_gs = []
    # for f in my_config.data['train_graph_path'][:1]:
    #     gs, labels = load_graphs(f)
    #     train_gs.extend(gs)
    #     del gs, labels
    
    # train_infos = []
    # for f in my_config.data['train_info_path'][:1]:
    #     infos = json.load(open(f))
    #     train_infos.extend(infos)
    #     del infos
    # train_infos = [e[0] for e in train_infos]

    # with open(my_config.data['test_path'],'r') as f:
    #     test_data = json.load(f)
    #     test_data = test_data['data']
    # with open(my_config.data['valid_path'],'r') as f:
    #     valid_data = json.load(f)
    # import pdb; pdb.set_trace()
    # train_data = list(zip(train_gs, train_infos))
    # train_data = [[e[4],e[0]] for e in train_data if len(e[4]) < my_config.data['max_src_len']]
    # test_data = [[e[4],e[0]] for e in test_data if len(e[4]) < my_config.data['max_src_len']]

    # load vocab
    with open(my_config.data['src_vocab_path'],'rb') as f:
        src_vocab = pickle.load(f)
    with open(my_config.data['tgt_vocab_path'],'rb') as f:
        tgt_vocab = pickle.load(f)

    node_edge_dict = json.load(open(my_config.data['node_edge_dict'], 'r'))


    mylogger.write("src_vocab: " + str(src_vocab))
    mylogger.write("tgt_vocab: " + str(tgt_vocab))

    # build model
    if my_config.model['name'] == 'hgt_copy_transformer':
        '''
        (self, src_vocab, tgt_vocab, embedding_dim=256,
                 hidden_size=2048, nlayers=8, use_cuda=True, dropout=0.2, nhead=8, node_edge_dict=None):
        '''
        model = HGTCopyTransformer(src_vocab, tgt_vocab, my_config.model['embed_size'],
                                my_config.model['hidden_size'], my_config.model['nlayers'],
                                my_config.use_cuda,
                                my_config.model['dropout'], 
                                my_config.model['nhead'],
                                node_edge_dict=node_edge_dict,
                                max_len=my_config.data['max_src_len'])
    elif my_config.model['name'] == 'hgt_copy_transformer_subtoken':
        model = HGTCopyTransformer_only_subtoken(src_vocab, tgt_vocab, my_config.model['embed_size'],
                                my_config.model['hidden_size'], my_config.model['nlayers'],
                                my_config.use_cuda,
                                my_config.model['dropout'], 
                                my_config.model['nhead'],
                                node_edge_dict=node_edge_dict,
                                max_len=my_config.data['max_src_len'])
    else:
        raise ValueError("model name {} not supported".format(my_config.model['name']))

    mylogger.write("model: " + str(model))

    
    
    model.to(my_config.device)
    # import os
    
    # torch.distributed.init_process_group(
    #     backend='nccl', init_method='env://')
    # torch.cuda.set_device(local_rank)
    # model = DataParallel(model, find_unused_parameters=True)
    #                      device_ids=[local_rank])
    
    # load model
    if my_config.load_model and os.path.exists(my_config.save['save_model_path']):
        model.load_state_dict(torch.load(my_config.save['save_model_path']))
        mylogger.write("load model from {}".format(
            my_config.save['save_model_path']))

    # model = DataParallel(model)


    
    
    
    # optimizer

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': my_config.optimizer['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=my_config.optimizer['lr'], eps=my_config.optimizer['adam_epsilon'])
    
    model.train()
    model.zero_grad()

    # choose loss function
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MultiMarginLoss()

    # loop over the dataset multiple times
    max_acc = 0
    
    # debug gradients
    # torch.autograd.set_detect_anomaly(True)
    if TRAIN:
        for epoch in range(my_config.optimizer['epochs']):
            # train_iter(train_data, model, optimizer, epoch)
            train_multi_iter(
                my_config.data['train_graph_path'][0:], my_config.data['train_info_path'][0:], model, optimizer, epoch)
            #torch.cuda.empty_cache()
            # test_iter(train_data[:1000], model, epoch, DEBUG= True)
            # #torch.cuda.empty_cache()
            # # acc, loss = test_iter(dev_dataloader, model, criterion, epoch)
            # # write_log(csv_log,str(epoch)+','+str(acc)+','+str(loss)+'\n')
            # # #torch.cuda.empty_cache()
            # test_iter(test_data, model, epoch, DEBUG = False, write_to_file=True)
            # #torch.cuda.empty_cache()
            if my_config.save_model:
                torch.save(model.state_dict(), my_config.save['save_model_path'] + '_' + str(epoch))
                # if acc > max_acc:
                    # max_acc = acc
                    # torch.save(model.state_dict(), my_config.path['save'] + '/model.pt')
        mylogger.write('Finished Training')

    else:
        mylogger.write('Only Testing...')
        test_iter_batch(my_config.data['test_graph_path'], my_config.data['test_info_path'],
                  model, 2021, DEBUG=False, write_to_file=True)


    
    
if __name__ == '__main__':
    main(TRAIN = my_config.TRAIN)
