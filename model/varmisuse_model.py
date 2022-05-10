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

    def forward(self, y_hat, y, masked_y_hat):
        # torch.log_softmax(y_hat,dim=-1) * masked_y_hat
        return self.loss(masked_log_softmax(y_hat, masked_y_hat, dim = -1), torch.argmax(y.long(), dim=-1))
class VarMisUseDecoder(torch.nn.Module):
    def __init__(self,
                 encoder_dim,
                 decoder_dim,
                 decoder_dp):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.decoder_dp = decoder_dp
        self.linear1 = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(self.decoder_dim)
        self.dp = nn.Dropout(self.decoder_dp)
        self.linear2 = nn.Linear(self.decoder_dim, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = torch.einsum('bld->bdl', x)
        x = self.bn(x)
        x = self.dp(x)
        x = torch.einsum('bdl->bld', x)
        x = self.linear2(x)
        return x

class HGTVarmisuse(torch.nn.Module):
    def __init__(self, src_vocab, embedding_dim=256,
                 encoder_hidden_size=2048, nlayer=8, use_cuda=True, dropout=0.2, nhead=8, node_edge_dict=None):
        super(HGTVarmisuse, self).__init__()
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
        self.decoder = VarMisUseDecoder(self.encoder_dim, self.decoder_dim, self.decoder_dp)
        
        self.loc_loss_func = CategoricalCrossEntropyLoss()
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
    def forward(self, src_batch_graph, identifier_nums):
        # import pdb;pdb.set_trace()
        # src_batch_graph = dgl.batch(src_graphs)
        # identifier_nums = [g.num_nodes('identifier') for g in src_graphs]
        encoder_graphs = self.encoder(src_batch_graph)
        
        # encoder_graph_list = dgl.unbatch(encoder_graphs)

        # import pdb; pdb.set_trace();

        # src_sents_idx = list(
        #                 encoder_graphs.nodes['identifier'].data['name'].split(identifier_nums))
        # src_sents_idx = [e.tolist() for e in src_sents_idx]
        # # src_sents_idx = [flatten_hete_ndata(
        #     # g, 'name', ntypes=['identifier', 'mod']).tolist()[:self.max_len] for g in encoder_graph_list]
        # src_sents = [[self.src_vocab.id2word[i]
        #               for i in sent] for sent in src_sents_idx]
        
        # # import pdb;pdb.set_trace()

        # lengths = [len(e) for e in src_sents]
        # src_mask = self.encoder.length_array_to_mask_tensor(lengths)
        # import pdb;pdb.set_trace()
        lengths = [e + 1 for e in identifier_nums]
        decoder_mask = self.encoder.length_array_to_mask_tensor(lengths)

        # [seq_len, hidden_size] * batch_size
        encoder_identifier_output = list(
            encoder_graphs.nodes['identifier'].data['h'].split(identifier_nums))
        encoder_mod_output = list(
            encoder_graphs.nodes['mod'].data['h'].split([1] * len(identifier_nums)))
        encoder_concat_output = [torch.cat((e1, e2), dim=0) for e1, e2 in zip(
            encoder_mod_output, encoder_identifier_output)]

        # loc_target
        loc_target_tensor_identifier = list(
            encoder_graphs.nodes['identifier'].data['loc_target'].long().split(identifier_nums))
        no_buggy = list(
            encoder_graphs.nodes['mod'].data['loc_target'].long().split([1] * len(identifier_nums)))
        loc_target_tensor = [torch.cat((e1, e2), dim=0).unsqueeze(-1) for e1, e2 in zip(
            no_buggy, loc_target_tensor_identifier)]

        # valid_mask
        valid_tensor_identifier = list(
            encoder_graphs.nodes['identifier'].data['valid_mask'].long().split(identifier_nums))
        valid_tensor_mod = list(
            encoder_graphs.nodes['mod'].data['valid_mask'].long().split([1] * len(identifier_nums)))
        valid_tensor = [torch.cat((e1, e2), dim=0).unsqueeze(-1) for e1, e2 in zip(
            valid_tensor_mod, valid_tensor_identifier)]


        # repair_target
        repair_target_identifier = list(
            encoder_graphs.nodes['identifier'].data['repair_target'].long().split(identifier_nums))
        repair_target_mod = list(
            encoder_graphs.nodes['mod'].data['repair_target'].long().split([1] * len(identifier_nums)))
        repair_target_tensor = [torch.cat((e1, e2), dim=0).unsqueeze(-1) for e1, e2 in zip(
            repair_target_mod, repair_target_identifier)]


        
        # import pdb;pdb.set_trace()
        encoder_concat_output = self.batch_different_length_tensor(
            encoder_concat_output)
        loc_target_tensor = self.batch_different_length_tensor(
            loc_target_tensor).squeeze()
        repair_target_tensor = self.batch_different_length_tensor(
            repair_target_tensor).squeeze()
        
        valid_tensor = self.batch_different_length_tensor(
            valid_tensor).squeeze()
        
        decoder_output = self.decoder(encoder_concat_output)
        # import pdb;pdb.set_trace()
        decoder_output += decoder_mask.unsqueeze(-1) * (-10000)

        loc_predictions = decoder_output[:, :, 0]

        loc_loss = self.loc_loss_func(loc_predictions, loc_target_tensor, valid_tensor)
        # loc_loss = loc_loss.mean()

        no_buggy = torch.stack(no_buggy).squeeze()



        # loc_equal = loc_predictions.argmax(dim = -1) == loc_target_tensor.argmax(dim =-1)

        # no_bug_pred_acc_num = (no_buggy * loc_equal).sum().item()
        # bug_loc_acc_num = ((1 - no_buggy) * loc_equal).sum().item()

        # import pdb;pdb.set_trace()
        repair_logits = decoder_output[:, :, 1]
        repair_logits_softmax = torch.softmax(repair_logits + (valid_tensor + 1e-45).log(), dim=-1)
        # repair_logits_softmax = masked_log_softmax(repair_logits, valid_tensor, dim = -1)
        # repair_logits = repair_logits * valid_tensor
        # repair_logits_softmax = torch.softmax(repair_logits, dim = -1)
        # import pdb;pdb.set_trace()
        
        repair_prob = repair_logits_softmax * repair_target_tensor
        repair_prob = repair_prob.sum(dim=-1)
        repair_loss = (1 - no_buggy) * -torch.log(repair_prob + 1e-9) / (1e-9 + (1 - no_buggy).sum(dim = -1))
        # repair_loss = repair_loss.mean()

        # repair_acc = (repair_prob > 0.5).float()
        # repair_acc_num = (repair_acc * (1 - no_buggy)).sum().item()

        return loc_loss, repair_loss
    

    def inference(self, src_batch_graph, identifier_nums):
        encoder_graphs = self.encoder(src_batch_graph)
        lengths = [e + 1 for e in identifier_nums]
        decoder_mask = self.encoder.length_array_to_mask_tensor(lengths)

        # [seq_len, hidden_size] * batch_size
        encoder_identifier_output = list(
            encoder_graphs.nodes['identifier'].data['h'].split(identifier_nums))
        encoder_mod_output = list(
            encoder_graphs.nodes['mod'].data['h'].split([1] * len(identifier_nums)))
        encoder_concat_output = [torch.cat((e1, e2), dim=0) for e1, e2 in zip(
            encoder_mod_output, encoder_identifier_output)]

        # loc_target
        loc_target_tensor_identifier = list(
            encoder_graphs.nodes['identifier'].data['loc_target'].long().split(identifier_nums))
        no_buggy = list(
            encoder_graphs.nodes['mod'].data['loc_target'].long().split([1] * len(identifier_nums)))
        loc_target_tensor = [torch.cat((e1, e2), dim=0).unsqueeze(-1) for e1, e2 in zip(
            no_buggy, loc_target_tensor_identifier)]

        # valid_mask
        valid_tensor_identifier = list(
            encoder_graphs.nodes['identifier'].data['valid_mask'].long().split(identifier_nums))
        valid_tensor_mod = list(
            encoder_graphs.nodes['mod'].data['valid_mask'].long().split([1] * len(identifier_nums)))
        valid_tensor = [torch.cat((e1, e2), dim=0).unsqueeze(-1) for e1, e2 in zip(
            valid_tensor_mod, valid_tensor_identifier)]

        # repair_target
        repair_target_identifier = list(
            encoder_graphs.nodes['identifier'].data['repair_target'].long().split(identifier_nums))
        repair_target_mod = list(
            encoder_graphs.nodes['mod'].data['repair_target'].long().split([1] * len(identifier_nums)))
        repair_target_tensor = [torch.cat((e1, e2), dim=0).unsqueeze(-1) for e1, e2 in zip(
            repair_target_mod, repair_target_identifier)]

        # import pdb;pdb.set_trace()
        encoder_concat_output = self.batch_different_length_tensor(
            encoder_concat_output)
        loc_target_tensor = self.batch_different_length_tensor(
            loc_target_tensor).squeeze()
        repair_target_tensor = self.batch_different_length_tensor(
            repair_target_tensor).squeeze()

        valid_tensor = self.batch_different_length_tensor(
            valid_tensor).squeeze()

        decoder_output = self.decoder(encoder_concat_output)
        # import pdb;pdb.set_trace()
        decoder_output += decoder_mask.unsqueeze(-1) * (-10000)

        loc_predictions = decoder_output[:, :, 0]

        loc_loss = self.loc_loss_func(
            loc_predictions, loc_target_tensor, valid_tensor)
        loc_loss = loc_loss.sum()

        no_buggy = torch.stack(no_buggy).squeeze()

        loc_equal = loc_predictions.argmax(dim = -1) == loc_target_tensor.argmax(dim =-1)
        # import pdb;pdb.set_trace()
        no_bug_pred_acc_num = (no_buggy * loc_equal).sum().item()
        bug_loc_acc_num = ((1 - no_buggy) * loc_equal).sum().item()

        # import pdb;pdb.set_trace()
        repair_logits = decoder_output[:, :, 1]
        repair_logits_softmax = torch.softmax(
            repair_logits + (valid_tensor + 1e-45).log(), dim=-1)
        repair_logits_softmax = masked_log_softmax(repair_logits, valid_tensor, dim = -1)
        repair_logits = repair_logits * valid_tensor
        repair_logits_softmax = torch.softmax(repair_logits, dim = -1)
        # import pdb;pdb.set_trace()

        repair_prob = repair_logits_softmax * repair_target_tensor
        repair_prob = repair_prob.sum(dim=-1)
        repair_loss = (1 - no_buggy) * -torch.log(repair_prob +
                                                  1e-9) / (1e-9 + (1 - no_buggy).sum(dim=-1))
        repair_loss = repair_loss.sum()

        repair_acc = (repair_prob > 0.5).float()
        repair_acc_num = (repair_acc * (1 - no_buggy)).sum().item()

        no_bug_num = no_buggy.sum().item()
        bug_num = (1 - no_buggy).sum().item()

        return (loc_loss, repair_loss), (no_bug_pred_acc_num, bug_loc_acc_num, repair_acc_num, no_bug_num, bug_num)
    


               


if __name__ == '__main__':
    sys.path.append('/home/zhangkechi/workspace/HGT-DGL/varmisuse_main')
    from varmisuse_main.varmisuse_dataloader import unit_test as dataloader_unit_test
    import pickle
    import json
    vocab_path = '/home/zhangkechi/workspace/HGT-DGL/data/varmisuse_great/step4/vocab.pkl'
    vocab = pickle.load(open(vocab_path, 'rb'))
    node_edge_dict_path = '/home/zhangkechi/workspace/HGT-DGL/data/varmisuse_great/step4/node_edge_dict.json'
    node_edge_dict = json.load(open(node_edge_dict_path, 'r'))

    model = HGTVarmisuse(vocab, node_edge_dict=node_edge_dict)
    dataloader = dataloader_unit_test()
    for i, batch in enumerate(dataloader):
        print(i)
        batched_graphs, all_identifier_nums = batch
        print(all_identifier_nums)
        loc_loss, repair_loss = model(batched_graphs, all_identifier_nums)
        loss = loc_loss.mean() + repair_loss.mean()
        loss.backward()

        (loc_loss, repair_loss), (no_bug_pred_acc_num, bug_loc_acc_num, repair_acc_num,
                                  no_bug_num, bug_num) = model.inference(batched_graphs, all_identifier_nums)
        print((loc_loss, repair_loss), (no_bug_pred_acc_num,
              bug_loc_acc_num, repair_acc_num, no_bug_num, bug_num))

        break

