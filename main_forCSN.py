import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

from model.seq2seq import CopyTransformer

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


def train_iter(dataset, model, optimizer, epoch):
    model.train()
    mylogger.write("start training...")
    bar = tqdm(total=len(dataset))
    running_loss = 0.0
    for batch_data in batch_iter(dataset, my_config.optimizer['bs'], True):
        bar.update(my_config.optimizer['bs'])
        # import pdb; pdb.set_trace()
        # exit()
        # inputs, labels = batch_data
        inputs = [e[0] for e in batch_data]
        labels = [e[1] for e in batch_data]
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        scores = model(inputs, labels)
        
        loss = -torch.sum(scores)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(
        #     model.parameters(), my_config.optim['max_grad_norm'])

        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()
        running_loss += loss.item()
        bar.set_description("epoch {} loss {}".format(epoch, loss.item()))
        # torch.cuda.empty_cache()

    mylogger.write("epoch {} loss {}".format(epoch, running_loss))



def test_iter(dataset, model, epoch, DEBUG = False, write_to_file = False):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    mylogger.write("start testing...")
    with torch.no_grad():
        bar = tqdm(total=len(dataset))
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_inputs = []
        for batch_data in batch_iter(dataset, my_config.optimizer['test_bs'], True):
            bar.update(my_config.optimizer['test_bs'])
            inputs = [e[0] for e in batch_data]
            labels = [e[1] for e in batch_data]
            for i,e in enumerate(inputs):
                outs = model.sample(e, None,
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


        
def main(TRAIN = True):
    set_seed(my_config.seed)

    # load data
    with open(my_config.data['train_path'],'r') as f:
        train_data = json.load(f)
        train_data = train_data['data']
    with open(my_config.data['test_path'],'r') as f:
        test_data = json.load(f)
        test_data = test_data['data']
    # with open(my_config.data['valid_path'],'r') as f:
    #     valid_data = json.load(f)
    # import pdb; pdb.set_trace()
    train_data = [[e[4],e[0]] for e in train_data if len(e[4]) < my_config.data['max_src_len']]
    test_data = [[e[4],e[0]] for e in test_data if len(e[4]) < my_config.data['max_src_len']]

    # load vocab
    with open(my_config.data['src_vocab_path'],'rb') as f:
        src_vocab = pickle.load(f)
    with open(my_config.data['tgt_vocab_path'],'rb') as f:
        tgt_vocab = pickle.load(f)

    mylogger.write("src_vocab: " + str(src_vocab))
    mylogger.write("tgt_vocab: " + str(tgt_vocab))

    # build model
    if my_config.model['name'] == 'copy_transformer':
        model = CopyTransformer(src_vocab, tgt_vocab, my_config.model['embed_size'], 
                                my_config.model['hidden_size'], my_config.model['nlayers'],
                                my_config.use_cuda,
                                my_config.model['dropout'], 
                                my_config.model['nhead'])
        '''
        (self, src_vocab, tgt_vocab, embedding_dim=256,
                 hidden_size=2048, nlayers=8, use_cuda = True, dropout = 0.2,nhead=8):
        '''
    else:
        raise ValueError("model name {} not supported".format(my_config.model['name']))

    mylogger.write("model: " + str(model))

    
    
    model.to(my_config.device)
    model = torch.nn.DataParallel(model)

    # load model

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

    # loop over the dataset multiple times
    max_acc = 0
    
    # debug gradients
    # torch.autograd.set_detect_anomaly(True)
    if TRAIN:
        for epoch in range(my_config.optimizer['epochs']):
            train_iter(train_data, model, optimizer, epoch)
            torch.cuda.empty_cache()
            test_iter(train_data[:1000], model, epoch, DEBUG= True)
            torch.cuda.empty_cache()
            # acc, loss = test_iter(dev_dataloader, model, criterion, epoch)
            # write_log(csv_log,str(epoch)+','+str(acc)+','+str(loss)+'\n')
            # torch.cuda.empty_cache()
            test_iter(test_data, model, epoch, DEBUG = False, write_to_file=True)
            torch.cuda.empty_cache()
            if my_config.save_model:
                torch.save(model.state_dict(), my_config.save['save_model_path'] + '_' + str(epoch))
                # if acc > max_acc:
                    # max_acc = acc
                    # torch.save(model.state_dict(), my_config.path['save'] + '/model.pt')
        mylogger.write('Finished Training')

    else:
        mylogger.write('Only Testing...')
        test_iter(test_data[:100], model, 2021, DEBUG = False, write_to_file=True)


    
    
if __name__ == '__main__':
    main(TRAIN = True)
