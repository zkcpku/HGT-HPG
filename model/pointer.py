# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class PointerNet(nn.Module):
    def __init__(self, query_vec_size, src_encoding_size, attention_type='affine'):
        super(PointerNet, self).__init__()

        assert attention_type in ('affine', 'dot_prod')
        if attention_type == 'affine':
            self.src_encoding_linear = nn.Linear(src_encoding_size, query_vec_size, bias=False)
        self.attention_type = attention_type

    def forward(self, src_encodings, src_token_mask, query_vec):
        """
        :param src_encodings: Variable(batch_size, src_sent_len, src_encoding_size)
        :param src_token_mask: Variable(batch_size, src_sent_len)
        :param query_vec: Variable(batch_size, tgt_sent_len, query_vec_size)
        :return: Variable(batch_size, src_sent_len, tgt_action_num)
        """

        # (batch_size, src_sent_len, query_vec_size)
        if self.attention_type == 'affine':
            src_encodings = self.src_encoding_linear(src_encodings)
        # src_encodings = src_encodings.unsqueeze(1)
        src_encodings = src_encodings

        # (batch_size, tgt_sent_len, query_vec_size)
        # q = query_vec.unsqueeze(3)
        q = query_vec
        # import pdb;pdb.set_trace();
        # (batch_size, tgt_sent_len, src_sent_len)
        weights = torch.einsum("abc,adc->abd", q, src_encodings)

        if src_token_mask is not None:
            src_token_mask = src_token_mask.unsqueeze(1).expand_as(weights)
            weights.data.masked_fill_(src_token_mask, -float('inf'))

        ptr_weights = F.softmax(weights, dim=-1)
        return ptr_weights
