import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import dgl


from utils.vocab import VocabEntry
from utils.nn_utils import to_input_variable
from model.hgt import HGTEncoder, HGTLayer, flatten_hete_ndata
from model.copy_transformer import EncoderTransformer, CopyDecoderTransformer




def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


class CategoricalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropyLoss, self).__init__()
        self.loss = nn.NLLLoss()

    def forward(self, y_hat, y_label, masked_y_hat):
        # torch.log_softmax(y_hat,dim=-1) * masked_y_hat
        return self.loss(masked_log_softmax(y_hat, masked_y_hat, dim=-1), y_label)


class ClassficationDecoder(torch.nn.Module):
    def __init__(self,
                 encoder_dim,
                 decoder_dim,
                 decoder_dp,
                 cls_num):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.decoder_dp = decoder_dp
        self.cls_num = cls_num
        self.linear1 = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.decoder_dim)
        self.dp = nn.Dropout(self.decoder_dp)
        self.linear2 = nn.Linear(self.decoder_dim, self.cls_num)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        # x = torch.einsum('bd->bd', x)
        x = self.bn(x)
        x = self.dp(x)
        # x = torch.einsum('bdl->bld', x)
        x = self.linear2(x)
        return x


class GlobalAttentionPooling_hete(torch.nn.Module):
    def __init__(self, gate_nn):
        super().__init__()
        self.gate_nn = gate_nn
    
    def flatten_hete_ndata(self, out_G, node_key = 'h'):
        if type(out_G.ndata[node_key]) is not torch.Tensor:
            return torch.cat([out_G.ndata[node_key][k] for k in out_G.ndata[node_key]])
        else:
            return out_G.ndata[node_key]

    def forward(self, G):
        feat = self.flatten_hete_ndata(G)
        gate = self.gate_nn(feat)
        gate = F.softmax(gate, dim=0)
        out_r = feat * gate
        readout = torch.sum(out_r, dim=0, keepdim=True)
        return readout

class GlobalAttentionPooling_on_feat(torch.nn.Module):
    def __init__(self, gate_nn):
        super().__init__()
        self.gate_nn = gate_nn
    def forward(self, feat):
        gate = self.gate_nn(feat)
        gate = F.softmax(gate, dim=1)
        out_r = feat * gate
        readout = torch.sum(out_r, dim=1, keepdim=True)
        return readout

class HGTClassfication(torch.nn.Module):
    def __init__(self, cls_num, src_vocab, embedding_dim=256,
                 encoder_hidden_size=2048, nlayer=8, use_cuda=True, dropout=0.2, nhead=8, node_edge_dict=None):
        super().__init__()
        self.cls_num = cls_num
        self.src_vocab = src_vocab
        self.src_vocab_size = len(src_vocab)

        self.encoder_embedding_dim = embedding_dim
        self.encoder_dim = embedding_dim
        self.decoder_dim = embedding_dim * 2

        self.encoder_hidden_size = encoder_hidden_size

        self.encoder_nlayers = nlayer

        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.encoder_dp = dropout
        self.decoder_dp = dropout

        self.encoder_nhead = nhead

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
        self.decoder = ClassficationDecoder(
            self.encoder_dim, self.decoder_dim, self.decoder_dp, self.cls_num)

        # self.global_attn = GlobalAttentionPooling_hete(gate_nn = torch.nn.Sequential(
        #     torch.nn.Linear(self.encoder_dim, 2 * self.encoder_dim),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(2 * self.encoder_dim, 1)
        # ))

        self.global_attn = GlobalAttentionPooling_on_feat(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(self.encoder_dim, 2 * self.encoder_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * self.encoder_dim, 1)
        ))

        self.loss_func = F.cross_entropy
        # self.loss_func = CategoricalCrossEntropyLoss()
        # https://huggingface.co/transformers/model_doc/bert.html?highlight=bertconfig#transformers.BertConfig
        self.initializer_range = 0.02
        self.init_weights()

    def init_weights(self):
        # self.encoder.to(self.device)
        # self.decoder.to(self.device)
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
            batch_tensor[i, :t.size(0), :] = t
        return batch_tensor.cuda() if self.use_cuda else batch_tensor

    # def forward(self, src_graphs, tgt_sents, src_inputs = None, tgt_inputs = None, out_key = ['identifier', '__ALL__'][1]):
    def forward(self, src_batch_graph, labels, node_nums):
        encoder_graphs = self.encoder(src_batch_graph)
        # import pdb; pdb.set_trace()
        # encoder_graphs = encoder_graphs.nodes['mod'].data['h']

        node_keys = list(node_nums[0].keys())[0]
        encoder_graphs = list(
            encoder_graphs.nodes[node_keys].data['h'].split([e[node_keys] for e in node_nums]))
        
        # decoder_mask = self.encoder.length_array_to_mask_tensor(
        #     node_nums[node_keys], valid_entry_has_mask_one=True)

        encoder_graphs = self.batch_different_length_tensor(
            encoder_graphs)

        encoder_graphs = self.global_attn(encoder_graphs).squeeze()
        logits = self.decoder(encoder_graphs)

        soft_logits = F.softmax(logits, dim=1)
        loss = self.loss_func(logits, labels)

        return loss, soft_logits




        # encoder_graphs = dgl.unbatch(encoder_graphs)
        # # import pdb; pdb.set_trace()
        # encoder_graphs = [self.global_attn(g) for g in encoder_graphs]
        
        # encoder_graphs = torch.cat(encoder_graphs)

        


if __name__ == '__main__':
    sys.path.append(
        '/home/zhangkechi/workspace/HGT-DGL/codenet_py_classification')
    from codenet_py_classification.classification_dataloader import unit_test as dataloader_unit_test
    import pickle
    import json
    vocab_path = '/home/zhangkechi/workspace/HGT-DGL/data/codenet/python/step3/vocab.pkl'
    vocab = pickle.load(open(vocab_path, 'rb'))
    node_edge_dict_path = '/home/zhangkechi/workspace/HGT-DGL/data/codenet/python/step3/node_edge_dict.json'
    node_edge_dict = json.load(open(node_edge_dict_path, 'r'))

    model = HGTClassfication(800, vocab, node_edge_dict=node_edge_dict)
    model = model.cuda()
    dataloader = dataloader_unit_test()
    for i, batch in enumerate(dataloader):
        print(i)
        batch_graph, target_sent, node_nums = batch
        batch_graph, target_sent = batch_graph.to(model.device), target_sent.to(model.device)

        loss, logits = model(batch_graph, target_sent,node_nums)
        loss.backward()


        print(loss, logits)
        break
