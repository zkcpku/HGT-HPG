import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.copy_transformer import EncoderTransformer, CopyDecoderTransformer
from model.hgt import HGTEncoder, HGTLayer, flatten_hete_ndata

from utils.nn_utils import to_input_variable
from utils.vocab import VocabEntry


# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class HGTCopyTransformer(torch.nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embedding_dim=256,
                 hidden_size=2048, nlayers=(4,8), use_cuda=True, dropout=0.2, nhead=8, node_edge_dict=None, max_len = 512):
        super(HGTCopyTransformer, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_vocab_size = len(src_vocab)
        self.tgt_vocab_size = len(tgt_vocab)

        self.encoder_embedding_dim = embedding_dim
        self.decoder_embedding_dim = embedding_dim
        self.encoder_dim = embedding_dim

        self.encoder_hidden_size = hidden_size
        self.decoder_hidden_size = hidden_size

        self.encoder_nlayers = nlayers[0]
        self.decoder_nlayers = nlayers[1]

        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.encoder_dp = dropout
        self.decoder_dp = dropout

        self.encoder_nhead = nhead
        self.decoder_nhead = nhead
        self.max_len = max_len

        # self.encoder_embedding = torch.nn.Embedding(self.src_vocab_size, self.encoder_embedding_dim)
        # self.decoder_embedding = torch.nn.Embedding(self.tgt_vocab_size, self.decoder_embedding_dim)

        # self.node_edge_dict = node_edge_dict
        self.node_dict = node_edge_dict['node']
        self.edge_dict = node_edge_dict['edge']

        # self.encoder = HGTEncoder(self.zero_graph ,self.encoder_embedding_dim)

        '''
        (self, G_node_dict, G_edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, len_src_vocab, device, use_norm = True):
        '''
        self.encoder = HGTEncoder(self.node_dict, self.edge_dict,
                                    self.encoder_embedding_dim, self.encoder_dim, self.encoder_dim,
                                    self.encoder_nlayers, self.encoder_nhead, self.src_vocab_size, self.device, use_norm=True)
                                    
        '''
        (self, action_size, encoder_dim,embedding_dim=256, hidden_size=2048, nlayers=6,
                 device=torch.device(
                     'cuda:0' if torch.cuda.is_available() else 'cpu'),
                 dp=0.3,
                 nhead=8)
        '''
        self.decoder = CopyDecoderTransformer(self.tgt_vocab_size, self.encoder_dim, self.decoder_embedding_dim,
                                              self.decoder_hidden_size, self.decoder_nlayers,
                                              self.device, self.decoder_dp, self.decoder_nhead)
        
        # https://huggingface.co/transformers/model_doc/bert.html?highlight=bertconfig#transformers.BertConfig
        self.initializer_range = 0.02
        self.init_weights()

    def init_weights(self):
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        # https://huggingface.co/transformers/_modules/transformers/modeling_utils.html#PreTrainedModel
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertModel
        """Initialize the weights"""
        if isinstance(module, torch.nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def batch_different_length_tensor(self, tensors):
        """
        Args:
            tensors: a list of tensors with different length
        Returns:
            a tensor with size (len(tensors), max(lengths), ...)
        """
        max_len = max(map(lambda t: t.size(0), tensors))
        batch_tensor = torch.zeros(len(tensors), max_len, tensors[0].size(-1))
        for i, t in enumerate(tensors):
            batch_tensor[i, :t.size(0),:] = t
        return batch_tensor.cuda() if self.use_cuda else batch_tensor

    # def forward(self, src_graphs, tgt_sents, src_inputs = None, tgt_inputs = None, out_key = ['identifier', '__ALL__'][1]):
    def forward(self, src_batch_graph, identifier_nums, tgt_sents, src_inputs=None, tgt_inputs=None, out_key=['identifier', '__ALL__'][1]):
        '''

        '''
        if tgt_inputs is None:
            tgt_inputs = to_input_variable(tgt_sents, self.tgt_vocab, cuda=self.use_cuda, append_boundary_sym=True).permute(1, 0)

        '''
        (self, G, out_key = '__ALL__'):
        '''
        # import pdb;pdb.set_trace()
        # src_batch_graph = dgl.batch(src_graphs)
        # identifier_nums = [g.num_nodes('identifier') for g in src_graphs]
        encoder_graphs = self.encoder(src_batch_graph)
        
        # encoder_graph_list = dgl.unbatch(encoder_graphs)

        # import pdb; pdb.set_trace();

        src_sents_idx = list(
                        encoder_graphs.nodes['identifier'].data['name'].split(identifier_nums))
        src_sents_idx = [e.tolist() for e in src_sents_idx]
        # src_sents_idx = [flatten_hete_ndata(
            # g, 'name', ntypes=['identifier', 'mod']).tolist()[:self.max_len] for g in encoder_graph_list]
        src_sents = [[self.src_vocab.id2word[i]
                      for i in sent][:self.max_len] for sent in src_sents_idx]
        
        # import pdb;pdb.set_trace()

        lengths = [len(e) for e in src_sents]
        src_mask = self.encoder.length_array_to_mask_tensor(lengths)
        tgt_token_copy_idx_mask, tgt_token_gen_mask = self.decoder.get_generate_and_copy_meta_tensor(
            src_sents, tgt_sents, self.src_vocab)

        # [seq_len, hidden_size] * batch_size
        encoder_output = list(
            encoder_graphs.nodes['identifier'].data['h'].split(identifier_nums))
        # import pdb;pdb.set_trace()
        encoder_output = [e[:self.max_len,:] for e in encoder_output]
        # encoder_output = [flatten_hete_ndata(
        #     g, 'h', ntypes=['identifier', 'mod'])[:self.max_len] for g in encoder_graph_list]
        encoder_output = self.batch_different_length_tensor(encoder_output)
        
        # each_out_key = ['identifier' if 'identifier' in encoder_graph_list[i].ntypes else '__ALL__' for i in range(len(encoder_graph_list))]
        
        # import pdb;pdb.set_trace()
        decoder_output = self.decoder(tgt_inputs, encoder_output, src_mask, None, tgt_token_copy_idx_mask, tgt_token_gen_mask)
        '''
        forward(self, targets, encoder_hidden, src_mask, tgt_mask, max_len = None):
        '''
        # loss = -torch.sum(decoder_output)
        return decoder_output
    
    def sample(self, src_graph, max_len=100, sample_size=5, mode='beam_search'):
        
        encoder_graph = self.encoder(src_graph)
        src_sent_idx = encoder_graph.nodes['identifier'].data['name'].tolist()
        src_sent = [self.src_vocab.id2word[i] for i in src_sent_idx][:self.max_len]

        # print(src_sent)

        lengths = [len(e) for e in [src_sent]]
        src_mask = self.encoder.length_array_to_mask_tensor(lengths)
        encoder_output = encoder_graph.nodes['identifier'].data['h'].unsqueeze(0)[:,:self.max_len,:]

        sample_output = self.decoder.sample(src_sent, encoder_output, src_mask,
                                            self.src_vocab, self.tgt_vocab,
                                            max_len=max_len, sample_size=sample_size, mode=mode)
        '''
        (self, src_sent, encoder_hidden, src_mask,src_vocab,
                    tgt_vocab,
                    max_len = 100, sample_size = 5, mode=['beam_search', 'sample'][0]):
        '''
        return sample_output

    def sample_batch(self, src_batch_graph, identifier_nums, max_len=100, sample_size=5, mode='beam_search'):
        bs = len(identifier_nums)
        encoder_graphs = self.encoder(src_batch_graph)
        src_sents_idx = list(
                        encoder_graphs.nodes['identifier'].data['name'].split(identifier_nums))
        src_sents_idx = [e.tolist() for e in src_sents_idx]
        src_sents = [[self.src_vocab.id2word[i]
                      for i in sent][:self.max_len] for sent in src_sents_idx]

        encoder_outputs = list(
            encoder_graphs.nodes['identifier'].data['h'].split(identifier_nums))
        # import pdb;pdb.set_trace()
        encoder_outputs = [e[:self.max_len,:] for e in encoder_outputs]
        sample_outputs = []
        for batch_idx in range(bs):
            src_sent = src_sents[batch_idx]
            lengths = [len(e) for e in [src_sent]]
            src_mask = self.encoder.length_array_to_mask_tensor(lengths)
            encoder_output = encoder_outputs[batch_idx].unsqueeze(0)
            sample_output = self.decoder.sample(src_sent, encoder_output, src_mask,
                                            self.src_vocab, self.tgt_vocab,
                                            max_len=max_len, sample_size=sample_size, mode=mode)
            
            sample_outputs.append(sample_output)
        
        return sample_outputs
        

class HGTCopyTransformer_only_subtoken(torch.nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embedding_dim=256,
                 hidden_size=2048, nlayers=(4,8), use_cuda=True, dropout=0.2, nhead=8, node_edge_dict=None, max_len = 512):
        super(HGTCopyTransformer_only_subtoken, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_vocab_size = len(src_vocab)
        self.tgt_vocab_size = len(tgt_vocab)

        self.encoder_embedding_dim = embedding_dim
        self.decoder_embedding_dim = embedding_dim
        self.encoder_dim = embedding_dim

        self.encoder_hidden_size = hidden_size
        self.decoder_hidden_size = hidden_size

        self.encoder_nlayers = nlayers[0]
        self.decoder_nlayers = nlayers[1]

        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.encoder_dp = dropout
        self.decoder_dp = dropout

        self.encoder_nhead = nhead
        self.decoder_nhead = nhead
        self.max_len = max_len

        # self.encoder_embedding = torch.nn.Embedding(self.src_vocab_size, self.encoder_embedding_dim)
        # self.decoder_embedding = torch.nn.Embedding(self.tgt_vocab_size, self.decoder_embedding_dim)

        # self.node_edge_dict = node_edge_dict
        self.node_dict = node_edge_dict['node']
        self.edge_dict = node_edge_dict['edge']

        # self.encoder = HGTEncoder(self.zero_graph ,self.encoder_embedding_dim)

        '''
        (self, G_node_dict, G_edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, len_src_vocab, device, use_norm = True):
        '''
        self.encoder = HGTEncoder(self.node_dict, self.edge_dict,
                                    self.encoder_embedding_dim, self.encoder_dim, self.encoder_dim,
                                    self.encoder_nlayers, self.encoder_nhead, self.src_vocab_size, self.device, use_norm=True)
                                    
        '''
        (self, action_size, encoder_dim,embedding_dim=256, hidden_size=2048, nlayers=6,
                 device=torch.device(
                     'cuda:0' if torch.cuda.is_available() else 'cpu'),
                 dp=0.3,
                 nhead=8)
        '''
        self.decoder = CopyDecoderTransformer(self.tgt_vocab_size, self.encoder_dim, self.decoder_embedding_dim,
                                              self.decoder_hidden_size, self.decoder_nlayers,
                                              self.device, self.decoder_dp, self.decoder_nhead)
        
        # https://huggingface.co/transformers/model_doc/bert.html?highlight=bertconfig#transformers.BertConfig
        self.initializer_range = 0.02
        self.init_weights()

    def init_weights(self):
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        # https://huggingface.co/transformers/_modules/transformers/modeling_utils.html#PreTrainedModel
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertModel
        """Initialize the weights"""
        if isinstance(module, torch.nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def batch_different_length_tensor(self, tensors):
        """
        Args:
            tensors: a list of tensors with different length
        Returns:
            a tensor with size (len(tensors), max(lengths), ...)
        """
        max_len = max(map(lambda t: t.size(0), tensors))
        batch_tensor = torch.zeros(len(tensors), max_len, tensors[0].size(-1))
        for i, t in enumerate(tensors):
            batch_tensor[i, :t.size(0),:] = t
        return batch_tensor.cuda() if self.use_cuda else batch_tensor

    # def forward(self, src_graphs, tgt_sents, src_inputs = None, tgt_inputs = None, out_key = ['identifier', '__ALL__'][1]):
    def forward(self, src_batch_graph, subtoken_nums, tgt_sents, src_inputs=None, tgt_inputs=None, out_key=['subtoken', '__ALL__'][1]):
        '''

        '''
        if tgt_inputs is None:
            tgt_inputs = to_input_variable(tgt_sents, self.tgt_vocab, cuda=self.use_cuda, append_boundary_sym=True).permute(1, 0)

        '''
        (self, G, out_key = '__ALL__'):
        '''
        # import pdb;pdb.set_trace()
        # src_batch_graph = dgl.batch(src_graphs)
        # subtoken_nums = [g.num_nodes('subtoken') for g in src_graphs]
        encoder_graphs = self.encoder(src_batch_graph)
        
        # encoder_graph_list = dgl.unbatch(encoder_graphs)

        # import pdb; pdb.set_trace();

        src_sents_idx = list(
                        encoder_graphs.nodes['subtoken'].data['name'].split(subtoken_nums))
        src_sents_idx = [e.tolist() for e in src_sents_idx]
        # src_sents_idx = [flatten_hete_ndata(
            # g, 'name', ntypes=['subtoken', 'mod']).tolist()[:self.max_len] for g in encoder_graph_list]
        src_sents = [[self.src_vocab.id2word[i]
                      for i in sent][:self.max_len] for sent in src_sents_idx]
        
        # import pdb;pdb.set_trace()

        lengths = [len(e) for e in src_sents]
        src_mask = self.encoder.length_array_to_mask_tensor(lengths)
        tgt_token_copy_idx_mask, tgt_token_gen_mask = self.decoder.get_generate_and_copy_meta_tensor(
            src_sents, tgt_sents, self.src_vocab)

        # [seq_len, hidden_size] * batch_size
        encoder_output = list(
            encoder_graphs.nodes['subtoken'].data['h'].split(subtoken_nums))
        # import pdb;pdb.set_trace()
        encoder_output = [e[:self.max_len,:] for e in encoder_output]
        # encoder_output = [flatten_hete_ndata(
        #     g, 'h', ntypes=['subtoken', 'mod'])[:self.max_len] for g in encoder_graph_list]
        encoder_output = self.batch_different_length_tensor(encoder_output)
        
        # each_out_key = ['subtoken' if 'subtoken' in encoder_graph_list[i].ntypes else '__ALL__' for i in range(len(encoder_graph_list))]
        
        # import pdb;pdb.set_trace()
        decoder_output = self.decoder(tgt_inputs, encoder_output, src_mask, None, tgt_token_copy_idx_mask, tgt_token_gen_mask)
        '''
        forward(self, targets, encoder_hidden, src_mask, tgt_mask, max_len = None):
        '''
        # loss = -torch.sum(decoder_output)
        return decoder_output
    
    def sample(self, src_graph, max_len=100, sample_size=5, mode='beam_search'):
        
        encoder_graph = self.encoder(src_graph)
        src_sent_idx = encoder_graph.nodes['subtoken'].data['name'].tolist()
        src_sent = [self.src_vocab.id2word[i] for i in src_sent_idx][:self.max_len]

        # print(src_sent)

        lengths = [len(e) for e in [src_sent]]
        src_mask = self.encoder.length_array_to_mask_tensor(lengths)
        encoder_output = encoder_graph.nodes['subtoken'].data['h'].unsqueeze(0)[:,:self.max_len,:]

        sample_output = self.decoder.sample(src_sent, encoder_output, src_mask,
                                            self.src_vocab, self.tgt_vocab,
                                            max_len=max_len, sample_size=sample_size, mode=mode)
        '''
        (self, src_sent, encoder_hidden, src_mask,src_vocab,
                    tgt_vocab,
                    max_len = 100, sample_size = 5, mode=['beam_search', 'sample'][0]):
        '''
        return sample_output

    def sample_batch(self, src_batch_graph, subtoken_nums, max_len=100, sample_size=5, mode='beam_search'):
        bs = len(subtoken_nums)
        encoder_graphs = self.encoder(src_batch_graph)
        src_sents_idx = list(
                        encoder_graphs.nodes['subtoken'].data['name'].split(subtoken_nums))
        src_sents_idx = [e.tolist() for e in src_sents_idx]
        src_sents = [[self.src_vocab.id2word[i]
                      for i in sent][:self.max_len] for sent in src_sents_idx]

        encoder_outputs = list(
            encoder_graphs.nodes['subtoken'].data['h'].split(subtoken_nums))
        # import pdb;pdb.set_trace()
        encoder_outputs = [e[:self.max_len,:] for e in encoder_outputs]
        sample_outputs = []
        for batch_idx in range(bs):
            src_sent = src_sents[batch_idx]
            lengths = [len(e) for e in [src_sent]]
            src_mask = self.encoder.length_array_to_mask_tensor(lengths)
            encoder_output = encoder_outputs[batch_idx].unsqueeze(0)
            sample_output = self.decoder.sample(src_sent, encoder_output, src_mask,
                                            self.src_vocab, self.tgt_vocab,
                                            max_len=max_len, sample_size=sample_size, mode=mode)
            
            sample_outputs.append(sample_output)
        
        return sample_outputs
               


if __name__ == '__main__':
    import json
    import pickle
    from dgl.data.utils import load_graphs
    # from torch.nn.parallel import DistributedDataParallel as DataParallel
    from torch.nn import DataParallel

    # load vocab
    tgt_vocab_path = '/home/zhangkechi/workspace/HGT-DGL/data/ogbg_code/seq2seq/tgt_vocab.pt'
    tgt_vocab = pickle.load(open(tgt_vocab_path, 'rb'))
    with open('/home/zhangkechi/workspace/HGT-DGL/data/ogbg_code/graph2seq/hete_no_subtoken/name_vocab.pkl', 'rb') as f:
        name_vocab = pickle.load(f)
    with open('/home/zhangkechi/workspace/HGT-DGL/data/ogbg_code/graph2seq/hete_no_subtoken/type_vocab.pkl', 'rb') as f:
        type_vocab = pickle.load(f)
    with open('/home/zhangkechi/workspace/HGT-DGL/data/ogbg_code/graph2seq/hete_no_subtoken/field_vocab.pkl', 'rb') as f:
        field_vocab = pickle.load(f)
    node_edge_dict = json.load(open('/home/zhangkechi/workspace/HGT-DGL/data/ogbg_code/graph2seq/hete_no_subtoken/node_edge_dict.json', 'r'))

    # load mini-batch data
    mini_batch_path = '/home/zhangkechi/workspace/HGT-DGL/data/ogbg_code/graph2seq/hete_no_subtoken/minibatch_graph.json_'
    mini_batch_info_path = '/home/zhangkechi/workspace/HGT-DGL/data/ogbg_code/graph2seq/hete_no_subtoken/minibatch_info.json_'
    gs, labels = load_graphs(mini_batch_path)
    infos = json.load(open(mini_batch_info_path))

    print(len(gs))
    print([g.number_of_nodes() for g in gs])
    # import pdb;pdb.set_trace();
    # print([e[0] for e in infos][:10])
    # print(gs[0].nodes['identifier'].data['name'])
    # print(gs[0].nodes['__all__'])
    # import pdb;pdb.set_trace();
    # init model
    '''
    (self, src_vocab, tgt_vocab, embedding_dim=256,
                 hidden_size=2048, nlayers=8, use_cuda=True, dropout=0.2, nhead=8, node_edge_dict=None)
    '''
    embedding_dim = 128
    hidden_size = 1024
    nlayers = (4,6)
    use_cuda = True
    dropout = 0.2
    nhead = 8
    batch_size = 32
    model = HGTCopyTransformer(name_vocab, tgt_vocab, embedding_dim, hidden_size, nlayers, use_cuda, dropout, nhead, node_edge_dict)
    model.to(torch.device('cuda'))
    # torch.distributed.init_process_group(backend='nccl')
    # model = DataParallel(model)
    # print(model)

    gs = gs[:batch_size]
    gs_num_nodes = [g.num_nodes('identifier') for g in gs]
    gs_batch = dgl.batch(gs)
    tgt_sents = [e[0] for e in infos[:batch_size]]
    
    # forward(self, src_graphs, tgt_sents, src_inputs = None, tgt_inputs = None, out_key = ['identifier', '__ALL__'][0]):
    scores = model(gs_batch, gs_num_nodes, tgt_sents)
    print(scores)
    loss = -torch.sum(scores)
    # print(loss)
    loss.backward()
    print(loss)

    # sample(self, src_graph, max_len=100, sample_size=5, mode='beam_search'):
    if isinstance(model, DataParallel):
        model = model.module
    model.eval()
    sample_input_graph = gs[0]
    sample_output = model.sample(sample_input_graph, max_len=100, sample_size=5, mode='beam_search')
    print(sample_output)

    sample_input_graphs = gs[:10]
    sample_input_num_nodes = [g.num_nodes('identifier') for g in sample_input_graphs]
    sample_input_graphs_batch = dgl.batch(sample_input_graphs)
    sample_output = model.sample_batch(sample_input_graphs_batch, sample_input_num_nodes, max_len=100, sample_size=5, mode='beam_search')
    print(sample_output)
        

    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False) as prof:
    #     outputs = model(batch_gs, tgt_sents)
    # print(prof.table())

    # import pdb; pdb.set_trace()


