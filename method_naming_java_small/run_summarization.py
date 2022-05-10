import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

from summarization_config import *
import utils.nn_utils as nn_utils
from utils.nn_utils import batch_iter
from utils.vocab import VocabEntry, Vocab
from utils.log_utils import logWriter, write_log

from utils.eval_utils import *
# from dgl.data import DGLDataset
# from dgl import save_graphs, load_graphs
# from dgl.data.utils import makedirs, save_info, load_info
from transformers import AdamW, get_linear_schedule_with_warmup

from model.graph2seq import HGTCopyTransformer, HGTCopyTransformer_only_subtoken

from summarization_dataloader import SingeJsonDataset,get_dataset,get_collate_fn

# torch.multiprocessing.set_sharing_strategy('file_system')
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


def train_epoch(dataloader, model, optimizer, epoch):
    mylogger.write("start training epoch {}...".format(str(epoch)))
    t1 = time.time()
    running_loss = 0.0
    model.train()
    bar = tqdm(total=len(dataloader))
    for i, batch_sample in enumerate(dataloader):
        try:
            bar.update(1)
            batch_graph, target_sent, all_subtoken_nums = batch_sample
            optimizer.zero_grad()
            scores = model(batch_graph, all_subtoken_nums, target_sent)
            loss = -torch.sum(scores)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            this_time_loss = float(loss.item())
            running_loss += this_time_loss
            bar.set_description(
                "epoch {} loss {}".format(epoch, this_time_loss))

            # torch.cuda.empty_cache()

        except Exception as e:
            mylogger.write(e)
            if my_config.save_model:
                torch.save(model.state_dict(),
                           my_config.save['save_model_path'] + '_' + str(epoch) + '_' + 'error')
            # mylogger.write(batch_data)
            # torch.cuda.empty_cache()
            continue
    # torch.cuda.empty_cache()
    t2 = time.time()
    mylogger.write("train epoch {} cost {}".format(str(epoch), t2-t1))

    mylogger.write("epoch {} loss {}".format(
        epoch, running_loss / (len(dataloader) // my_config.optimizer['bs'])))


def test_epoch(dataloader, model, epoch, DEBUG=False, write_to_file=False):
    if isinstance(model, DataParallel):
        model = model.module
    t1 = time.time()
    model.eval()
    # mylogger.write()
    mylogger.write("start testing...")
    with torch.no_grad():
        bar = tqdm(total=len(dataloader))
        running_loss = 0.0
        all_labels = []
        all_preds = []
        # all_inputs = []
        for i, batch_sample in enumerate(dataloader):
            bar.update(1)
            batch_graph, target_sent, all_subtoken_nums = batch_sample

            outs_batch = model.sample_batch(batch_graph, all_subtoken_nums,
                                            max_len=my_config.sample['decode_max_time_step'],
                                            sample_size=my_config.sample['sample_size'],
                                            mode=my_config.sample['mode'])
            # all_inputs.extend(e)
            all_labels.extend(list(target_sent))
            all_preds.extend(outs_batch)

            if DEBUG:
                break  # only test one batch
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
            for i, e in enumerate(all_labels):  # for debug
                # mylogger.write(all_inputs[i])
                mylogger.write(e)
                mylogger.write(all_preds[i])

    return test_rst


def main(TRAIN=True):
    set_seed(my_config.seed)

    # load vocab
    with open(my_config.data['src_vocab_path'], 'rb') as f:
        src_vocab = pickle.load(f)
    with open(my_config.data['tgt_vocab_path'], 'rb') as f:
        tgt_vocab = pickle.load(f)

    zero_dict = pickle.load(open(my_config.data['zero_dict'], 'rb'))

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
        raise ValueError("model name {} not supported".format(
            my_config.model['name']))

    mylogger.write("model: " + str(model))

    model.to(my_config.device)
    # import os

    # torch.distributed.init_process_group(
    #     backend='nccl', init_method='env://')
    # torch.cuda.set_device(local_rank)
    # model = DataParallel(model, find_unused_parameters=True)
    #                      device_ids=[local_rank])

    # load model
    # if my_config.load_model and os.path.exists(my_config.save['save_model_path']):
    #     model.load_state_dict(torch.load(my_config.save['save_model_path']))
    #     mylogger.write("load model from {}".format(
    #         my_config.save['save_model_path']))

    model = DataParallel(model)
    if my_config.load_model and os.path.exists(my_config.save['save_model_path']):
        model.load_state_dict(torch.load(my_config.save['save_model_path']))
        mylogger.write("load model from {}".format(
            my_config.save['save_model_path']))

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
    
    # load data
    t1 = time.time()
    collate_fn = get_collate_fn(src_vocab, zero_dict, 'subtoken' if not hasattr(
        my_config, 'split_node_type') else my_config.split_node_type, max_tgt_len = my_config.data['max_tgt_len'])
    if TRAIN:
        dataset = get_dataset(my_config.data['train_path'])
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=my_config.optimizer['bs'], shuffle=True, collate_fn=collate_fn, num_workers=5, prefetch_factor=10)
    else:
        dataset = get_dataset(my_config.data['test_path'])
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=my_config.optimizer['test_bs'], shuffle=False, collate_fn=collate_fn, num_workers=5, prefetch_factor=10)
    t2 = time.time()
    mylogger.write("load data cost {}".format(t2 - t1))

    # loop over the dataset multiple times
    max_acc = 0

    # debug gradients
    # torch.autograd.set_detect_anomaly(True)
    if TRAIN:
        for epoch in range(my_config.optimizer['start_epoch'],my_config.optimizer['epochs']):
            # train_iter(train_data, model, optimizer, epoch)
            train_epoch(dataloader, model, optimizer, epoch)
            # torch.cuda.empty_cache()
            # test_iter(train_data[:1000], model, epoch, DEBUG= True)
            # # torch.cuda.empty_cache()
            # # acc, loss = test_iter(dev_dataloader, model, criterion, epoch)
            # # write_log(csv_log,str(epoch)+','+str(acc)+','+str(loss)+'\n')
            # # # torch.cuda.empty_cache()
            # test_iter(test_data, model, epoch, DEBUG = False, write_to_file=True)
            # # torch.cuda.empty_cache()
            if my_config.save_model:
                torch.save(model.state_dict(),
                           my_config.save['save_model_path'] + '_' + str(epoch))
                # if acc > max_acc:
                # max_acc = acc
                # torch.save(model.state_dict(), my_config.path['save'] + '/model.pt')
        mylogger.write('Finished Training')

    else:
        mylogger.write('Only Testing...')
        test_epoch(dataloader, model, 2021, DEBUG=False, write_to_file=True)


if __name__ == '__main__':
    main(TRAIN=my_config.TRAIN)
