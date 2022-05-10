import os
import sys
from typing import Set
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from model.pointer import PointerNet

from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

from utils.nn_utils import generate_square_subsequent_mask

class EncoderDecoderTransformer(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder):
        super(EncoderDecoderTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = nn.Linear(
            self.decoder.transformer_dim, self.decoder.action_size, device=self.encoder.device)
        
    def forward(self, src, tgt, src_lengths):
        """Take in and process masked src and target sequences."""
        encoder_output = self.encoder(src, src_lengths)
        src_mask = self.encoder.length_array_to_mask_tensor(src_lengths)
        decoder_output = self.decoder(tgt, None, encoder_output, src_mask, None)
        pred = self.generator(decoder_output)
        return pred

    def sample(self, src , tgt_sos, src_lengths, max_len=100):
        """
        Samples a sequence of actions from the model.
        
        Args:
            tgt_sos: The start of sequence token. (batch_size, 1)
        """
        encoder_hidden = self.encoder(src, src_lengths)
        src_mask = self.encoder.length_array_to_mask_tensor(src_lengths)

        

        preds_idx = [tgt_sos]
        outputs = []
        for i in range(max_len):
            targets = torch.stack(preds_idx, dim=-1)
            target_token_emb = self.decoder.embedding(
                targets) * math.sqrt(self.decoder.embedding_dim)
            # seq, batch
            target_token_emb = target_token_emb.permute(1, 0, 2)
            target_pos_emb = self.decoder.pos_embedding(target_token_emb)
            # batch, seq
            target_pos_emb = target_pos_emb.permute(1, 0, 2)

            output = self.decoder.forward_step(
                target_pos_emb, encoder_hidden, src_mask)

            pred = self.generator(output)
            pred_idx = pred.argmax(dim=-1)
            preds_idx.append(pred_idx)

        return preds_idx


class PositionalEncodingSin(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncodingSin, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # import pdb; pdb.set_trace()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EncoderTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=2048, nlayers=6,
                 device=torch.device(
                     'cuda:0' if torch.cuda.is_available() else 'cpu'),
                 dp=0.3,
                 nhead=8):
        super(EncoderTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout_p = dp
        self.device = device
        self.nhead = nhead
        self.num_encoder_layers = nlayers
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = PositionalEncodingSin(
            embedding_dim, dropout=dp, max_len=5000)
        encoder_layers = nn.TransformerEncoderLayer(
            embedding_dim, nhead, hidden_size, dp)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)

    def length_array_to_mask_tensor(self, length_array, cuda=True, valid_entry_has_mask_one=False):
        max_len = max(length_array)
        batch_size = len(length_array)

        mask = np.zeros((batch_size, max_len), dtype=np.uint8)
        for i, seq_len in enumerate(length_array):
            if valid_entry_has_mask_one:
                mask[i][:seq_len] = 1
            else:
                mask[i][seq_len:] = 1

        # mask = torch.ByteTensor(mask)
        mask = torch.BoolTensor(mask)
        return mask.cuda() if cuda else mask

    def forward(self, input, src_mask, lengths = None):
        """
        Args:
            input: [batch_size, seq_len]
            src_mask: [batch_size, seq_len]
            lengths: list of [length_1, length_2, ...]
        
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # batch, seq
        token_emb = self.embedding(input) * math.sqrt(self.embedding_dim)
        # seq, batch
        token_emb = token_emb.permute(1, 0, 2)
        pos_emb = self.pos_embedding(token_emb)

        src_key_padding_mask = src_mask
        if src_mask is None:
            src_key_padding_mask = self.length_array_to_mask_tensor(lengths)
        
        # import pdb; pdb.set_trace()
        # print(src_key_padding_mask)
        encoder_output = self.transformer_encoder(
            pos_emb, src_key_padding_mask=src_key_padding_mask)

        # batch, seq, hidden
        encoder_output = encoder_output.permute(1, 0, 2)
    
        return encoder_output

class DecoderTransformer(nn.Module):
    def __init__(self, action_size, encoder_dim,embedding_dim=256, hidden_size=2048, nlayers=6,
                 device=torch.device(
                     'cuda:0' if torch.cuda.is_available() else 'cpu'),
                 dp=0.3,
                 nhead=8):
        super(DecoderTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.action_size = action_size
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout_p = dp
        self.device = device
        self.nhead = nhead
        self.num_decoder_layers = nlayers
        self.device = device

        self.embedding = nn.Embedding(action_size, embedding_dim)
        self.pos_embedding = PositionalEncodingSin(
            embedding_dim, dropout=dp, max_len=5000)


        self.use_parent_action = False
        if self.use_parent_action:
            self.transformer_dim = encoder_dim + embedding_dim
        else:
            self.transformer_dim = embedding_dim
        
        decoder_layers = nn.TransformerDecoderLayer(
            self.transformer_dim, nhead, hidden_size, dp)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layers, nlayers)

        self.init_weights()

    def init_weights(self):
        pass
        # initrange = 0.1
        # nn.init.uniform_(self.embedding.weight, -initrange, initrange)

    def forward_step(self, history_embed, encoder_hidden, src_mask):
        """
        Args:
            history_embed: [batch_size, seq_len, embed_dim]
            encoder_hidden: [batch_size, seq_len, hidden_size]
            src_mask: [batch_size, seq_len]
        """
        # import pdb;pdb.set_trace();
        history_embed = history_embed.permute(1, 0, 2)
        encoder_hidden = encoder_hidden.permute(1, 0, 2)
        # seq, batch, hidden
        output = self.transformer_decoder(
            history_embed, encoder_hidden, memory_key_padding_mask=src_mask)
        
        output = output[-1]
        # batch, hidden
        return output

    def forward(self, targets, encoder_hidden, src_mask, tgt_mask, max_len = None):
        """
        Args:
            targets: [batch_size, seq_len]
            encoder_hidden: [batch_size, seq_len, hidden_size]
            src_mask: [batch_size, seq_len]
            tgt_mask: [batch_size, seq_len]
        
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """

        if tgt_mask is None:
            tgt_mask = torch.ones(targets.size()).to(self.device)
        if max_len is None:
            max_len = tgt_mask.size(-1)
        B,L = targets.shape
        H = self.hidden_size
        if self.use_parent_action:
            concat_targets = targets
            targets, parents = concat_targets
            assert parents is not None
            parents_emb = self.embedding(parents)
            target_token_emb = torch.cat([self.embedding(targets), parents_emb], dim=-1) * math.sqrt(self.embedding_dim * 2)
        else:
            target_token_emb = self.embedding(targets) * math.sqrt(self.embedding_dim)
        # seq, batch
        target_token_emb = target_token_emb.permute(1, 0, 2)
        target_pos_emb = self.pos_embedding(target_token_emb)
        # batch, seq
        target_pos_emb = target_pos_emb.permute(1, 0, 2)


        outputs = []
        for i in range(max_len):
            history_embed = target_pos_emb[:, :(i+1), :]
            output = self.forward_step(history_embed, encoder_hidden, src_mask)
            outputs.append(output.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs



class CopyDecoderTransformer(nn.Module):
    def __init__(self, action_size, encoder_dim,embedding_dim=256, hidden_size=2048, nlayers=6,
                 device=torch.device(
                     'cuda:0' if torch.cuda.is_available() else 'cpu'),
                 dp=0.3,
                 nhead=8):
        super(CopyDecoderTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.action_size = action_size
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout_p = dp
        self.device = device
        self.nhead = nhead
        self.num_decoder_layers = nlayers
        self.device = device

        self.embedding = nn.Embedding(action_size, embedding_dim)
        self.pos_embedding = PositionalEncodingSin(
            embedding_dim, dropout=dp, max_len=5000)


        self.use_parent_action = False
        if self.use_parent_action:
            self.transformer_dim = encoder_dim + embedding_dim
        else:
            self.transformer_dim = embedding_dim
        
        decoder_layers = nn.TransformerDecoderLayer(
            self.transformer_dim, nhead, hidden_size, dp)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layers, nlayers)

        self.generator = nn.Linear(
            self.transformer_dim, self.action_size)


        # copy or generate classifier
        self.tgt_token_predictor = nn.Linear(self.transformer_dim, 2)

        self.pointer_network = PointerNet(self.transformer_dim, self.transformer_dim)

        self.tgt_start_with_sos = True

        self.init_weights()

    def init_weights(self):
        pass
        # initrange = 0.1
        # nn.init.uniform_(self.embedding.weight, -initrange, initrange)

    def embed(self, x):
        """
        Args:
            x: [batch_size, seq_len]
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        token_emb = self.embedding(x) * math.sqrt(self.embedding_dim)
        permute_emb = token_emb.permute(1, 0, 2)
        # seq, batch, embed_dim
        permute_pos_emb = self.pos_embedding(permute_emb)

        # batch, seq, embed_dim
        pos_emb = permute_pos_emb.permute(1, 0, 2)
        return pos_emb

    def forward_step(self, history_embed, encoder_hidden, src_mask):
        """
        Args:
            history_embed: [batch_size, seq_len, embed_dim]
            encoder_hidden: [batch_size, seq_len, hidden_size]
            src_mask: [batch_size, seq_len]

        Returns:
            output: [batch_size, hidden_size]
        """
        # import pdb;pdb.set_trace();
        history_embed = history_embed.permute(1, 0, 2)
        encoder_hidden = encoder_hidden.permute(1, 0, 2)
        # seq, batch, hidden
        # import pdb;pdb.set_trace();
        output = self.transformer_decoder(
            history_embed, encoder_hidden, memory_key_padding_mask=src_mask)
        
        output = output[-1]
        # batch, hidden
        return output
    def forward_steps(self, all_history_embed, encoder_hidden, src_mask):
        '''
        Args:
            all_history_embed: [batch_size, seq_len, embed_dim]
            encoder_hidden: [batch_size, seq_len, hidden_size]
            src_mask: [batch_size, seq_len]
        
        Returns:
            output: [batch_size, seq_len, hidden_size]
        '''
        tgt_seq_len = all_history_embed.size(1)
        all_history_embed = all_history_embed.permute(1, 0, 2)
        encoder_hidden = encoder_hidden.permute(1, 0, 2)

        tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(self.device)

        # seq, batch, hidden
        # import pdb;pdb.set_trace();
        output = self.transformer_decoder(
            all_history_embed, encoder_hidden, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
        # batch, seq, hidden
        return output

    def get_generate_and_copy_meta_tensor(self,src_codes, tgt_nls, src_vocab, use_cuda = True):
        tgt_nls = [['<s>'] + x + ['</s>'] for x in tgt_nls]
        max_time_step = max(len(tgt_nl) for tgt_nl in tgt_nls)
        max_src_len = max(len(src_code) for src_code in src_codes)
        batch_size = len(src_codes)

        tgt_token_copy_idx_mask = np.zeros(
            (max_time_step, batch_size, max_src_len), dtype='float32')
        tgt_token_gen_mask = np.zeros(
            (max_time_step, batch_size), dtype='float32')

        for t in range(max_time_step):
            for example_id, (src_code, tgt_nl) in enumerate(zip(src_codes, tgt_nls)):
                copy_pos = copy_mask = gen_mask = 0
                if t < len(tgt_nl):
                    tgt_token = tgt_nl[t]
                    copy_pos_list = [_i for _i, _token in enumerate(
                        src_code) if _token == tgt_token]
                    tgt_token_copy_idx_mask[t, example_id, copy_pos_list] = 1

                    # import pdb;pdb.set_trace()

                    gen_mask = 0
                    # we need to generate this token if (1) it's defined in the dictionary,
                    # or (2) it is an unknown word and not appear in the source side
                    if tgt_token in src_vocab:
                        gen_mask = 1
                    elif len(copy_pos_list) == 0:
                        gen_mask = 1

                    tgt_token_gen_mask[t, example_id] = gen_mask

        tgt_token_copy_idx_mask = Variable(
            torch.from_numpy(tgt_token_copy_idx_mask))
        tgt_token_gen_mask = Variable(torch.from_numpy(tgt_token_gen_mask))
        if use_cuda:
            tgt_token_copy_idx_mask = tgt_token_copy_idx_mask.cuda()
            tgt_token_gen_mask = tgt_token_gen_mask.cuda()

        return tgt_token_copy_idx_mask, tgt_token_gen_mask

    def forward(self, targets, encoder_hidden, src_mask, tgt_mask, tgt_token_copy_idx_mask, tgt_token_gen_mask, max_len = None):
        """
        Args:
            targets: [batch_size, seq_len]
            encoder_hidden: [batch_size, seq_len, hidden_size]
            src_mask: [batch_size, seq_len]
            tgt_mask: [batch_size, seq_len]
            tgt_token_copy_idx_mask: [seq_len, batch_size, max_src_len]
            tgt_token_gen_mask: [seq_len, batch_size]
        
        Returns:
            scores: [batch_size]
        """
        # import pdb;pdb.set_trace();
        if tgt_mask is None:
            tgt_mask = torch.zeros(targets.size()).to(self.device)
        if max_len is None:
            max_len = tgt_mask.size(-1)
        B,L = targets.shape
        H = self.hidden_size
        if self.use_parent_action:
            concat_targets = targets
            targets, parents = concat_targets
            assert parents is not None
            parents_emb = self.embedding(parents)
            target_token_emb = torch.cat([self.embedding(targets), parents_emb], dim=-1) * math.sqrt(self.embedding_dim * 2)
        else:
            target_token_emb = self.embedding(targets) * math.sqrt(self.embedding_dim)
        # seq, batch
        target_token_emb = target_token_emb.permute(1, 0, 2)
        target_pos_emb = self.pos_embedding(target_token_emb)
        # batch, seq
        target_pos_emb = target_pos_emb.permute(1, 0, 2)


        # outputs = []
        
        # for i in range(max_len-1):
        #     history_embed = target_pos_emb[:, :(i+1), :]
        #     # import pdb;pdb.set_trace();
        #     output = self.forward_step(history_embed, encoder_hidden, src_mask)
        #     outputs.append(output.unsqueeze(1))
        # outputs = torch.cat(outputs, dim=1)

        steps_outputs = self.forward_steps(target_pos_emb[:, :(max_len-1), :], encoder_hidden, src_mask)
        outputs = steps_outputs.permute(1, 0, 2)
        # copy

        # batch, seq, 2
        tgt_token_predictor = self.tgt_token_predictor(outputs)
        tgt_token_predictor = F.softmax(tgt_token_predictor, dim=-1)

        # generate prob for each token
        # (batch, seq, action_size)
        token_gen_prob = F.softmax(self.generator(outputs), dim=-1)
        # import pdb;pdb.set_trace();
        token_copy_prob = self.pointer_network(encoder_hidden, src_mask, outputs)

        # seq, batch
        tgt_token_idx = targets.permute(1, 0)
        if self.tgt_start_with_sos:
            tgt_token_idx = tgt_token_idx[1:]
            tgt_token_gen_mask = tgt_token_gen_mask[1:]
            tgt_token_copy_idx_mask = tgt_token_copy_idx_mask[1:]
        
        # seq, batch, action_size
        token_gen_prob = token_gen_prob.permute(1, 0, 2)
        # import pdb;pdb.set_trace();
        tgt_token_gen_prob = torch.gather(token_gen_prob, dim=2,
                                            index=tgt_token_idx.unsqueeze(2)).squeeze(2) * tgt_token_gen_mask

        # seq, batch, hidden
        token_copy_prob = token_copy_prob.permute(1, 0, 2)
        tgt_token_copy_prob = torch.sum(token_copy_prob * tgt_token_copy_idx_mask, dim=-1)

        # seq, batch, hidden
        tgt_token_predictor = tgt_token_predictor.permute(1, 0, 2)
        tgt_token_mask = torch.gt(tgt_token_gen_mask + tgt_token_copy_idx_mask.sum(dim=-1), 0.).float()
        tgt_token_prob = torch.log(tgt_token_predictor[:, :, 0] * tgt_token_gen_prob +
                                   tgt_token_predictor[:, :, 1] * tgt_token_copy_prob +
                                   1.e-7 * (1. - tgt_token_mask))

        tgt_token_prob = tgt_token_prob * tgt_token_mask

        scores = tgt_token_prob.sum(dim=0)

        return scores

    def sample_batch(self, src_sents, encoder_hiddens, src_masks, src_vocab, tgt_vocab, max_len = 100, sample_size = 5, mode = ['beam_search', 'sample'][0]):
        """
        Args:
            src_sents: [batch_size, seq_len]
            encoder_hiddens: [batch_size, seq_len, hidden_size]
            src_masks: [batch_size, seq_len]
            src_vocab: Vocab
            tgt_vocab: Vocab
            max_len: int
            sample_size: int
            mode: str
        Returns:
            samples: [batch_size, seq_len]
        """
        return
        bs = len(src_sents)
        assert mode in ['beam_search', 'sample']
        all_src_token_tgt_vocab_ids = [[tgt_vocab[token] for token in src_sent] for src_sent in src_sents]
        all_src_unk_pos_list = [[pos for pos, token_id in enumerate(src_token_tgt_vocab_ids) if
                            token_id == tgt_vocab.unk_id] for src_token_tgt_vocab_ids in all_src_token_tgt_vocab_ids]
        all_token_set = []
        for batch_idx in range(len(all_src_token_tgt_vocab_ids)):
            token_set = Set()
            for i, tid in enumerate(all_src_token_tgt_vocab_ids[batch_idx]):
                if tid in token_set:
                    all_src_token_tgt_vocab_ids[batch_idx][i] = -1
                else:
                    token_set.add(tid)
            all_token_set.append(token_set)

        t = 0
        eos_id = tgt_vocab['</s>']
        all_completed_hypotheses = [[] for _ in range(bs)]
        all_completed_hypothesis_scores = [[] for _ in range(bs)]

        if mode == 'beam_search':
            hypotheses = [['<s>'] for _ in range(bs)]
            hypotheses_word_ids = [[tgt_vocab['<s>']] for _ in range(bs)]
        else:
            hypotheses = [['<s>'] for _ in range(sample_size * bs)]
            hypotheses_word_ids = [[tgt_vocab['<s>']] for _ in range(sample_size * bs)]
        with torch.no_grad():
            hyp_scores = Variable(torch.FloatTensor(len(hypotheses)).to(self.device).zero_())

        done = [False for _ in range(bs)]

        while t < max_len:
            t += 1
            hyp_num = len(hypotheses)
            # encoder_hidden: [batch_size * beam, seq_len, hidden_size]
            expanded_src_encodings = encoder_hiddens.repeat_interleave(hyp_num, 0)
            # src_mask: [batch_size * beam, seq_len]
            expanded_src_masks = src_masks.repeat_interleave(hyp_num, 0)            
            
            # batch * beam, seq, hidden
            with torch.no_grad():
                already_generated_ys = Variable(torch.LongTensor(hypotheses_word_ids).to(self.device))

            ys_embedded = self.embed(already_generated_ys)

            # batch * beam, hidden
            decoder_output = self.forward_step(ys_embedded, expanded_src_encodings, expanded_src_masks)
            # batch * beam, 2
            tgt_token_predictor = F.softmax(self.tgt_token_predictor(decoder_output), dim=-1)
            # batch * beam, tgt_vocab_size
            token_gen_prob = F.softmax(self.generator(decoder_output), dim=-1)
            
            # ???need to check
            token_copy_prob = self.pointer_network(expanded_src_encodings, expanded_src_masks, decoder_output.unsqueeze(1)).squeeze(1)

            token_gen_prob = tgt_token_predictor[:, 0].unsqueeze(1) * token_gen_prob

            for batch_idx in range(bs):

                for token_pos, token_vocab_id in enumerate(all_src_token_tgt_vocab_ids[batch_idx]):
                    if token_vocab_id != -1 and token_vocab_id != tgt_vocab.unk_id:
                        p_copy = tgt_token_predictor[:, 1] * token_copy_prob[:, token_pos]
                        token_gen_prob[:, token_vocab_id] = token_gen_prob[:, token_vocab_id] + p_copy
                src_unk_pos_list = all_src_unk_pos_list[batch_idx]

                # second, add the probability of copying the most probable unk word
                gentoken_new_hyp_unks = []
                if src_unk_pos_list:
                    for hyp_id in range(hyp_num):
                        unk_pos = token_copy_prob[hyp_id][src_unk_pos_list].data.cpu().numpy().argmax()
                        unk_pos = src_unk_pos_list[unk_pos]
                        token = src_sents[batch_idx][unk_pos]
                        gentoken_new_hyp_unks.append(token)

                        unk_copy_score = tgt_token_predictor[hyp_id, 1] * token_copy_prob[hyp_id, unk_pos]
                        token_gen_prob[hyp_id, tgt_vocab.unk_id] = unk_copy_score

                live_hyp_num = sample_size - len(all_completed_hypotheses[batch_idx])

            if mode == 'beam_search':
                log_token_gen_prob = torch.log(token_gen_prob)
                new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(token_gen_prob) + log_token_gen_prob).view(-1)
                top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=live_hyp_num)
                # import pdb; pdb.set_trace();
                # prev_hyp_ids = (top_new_hyp_pos // len(tgt_vocab)).cpu().data
                prev_hyp_ids = (torch.div(top_new_hyp_pos, len(tgt_vocab), rounding_mode='floor')).cpu().data
                word_ids = (top_new_hyp_pos % len(tgt_vocab)).cpu().data
                top_new_hyp_scores = top_new_hyp_scores.cpu().data
            else:
                word_ids = torch.multinomial(token_gen_prob, num_samples=1)
                prev_hyp_ids = range(live_hyp_num)
                top_new_hyp_scores = hyp_scores + torch.log(torch.gather(token_gen_prob, dim=1, index=word_ids)).squeeze(1)
                top_new_hyp_scores = top_new_hyp_scores.cpu().data
                word_ids = word_ids.view(-1).cpu().data

            new_hypotheses = []
            new_hypotheses_word_ids = []
            live_hyp_ids = []
            new_hyp_scores = []
            for prev_hyp_id, word_id, new_hyp_score in zip(prev_hyp_ids, word_ids, top_new_hyp_scores):
                # import pdb;pdb.set_trace()
                word_id = word_id.item()
                if word_id == eos_id:
                    hyp_tgt_words = hypotheses[prev_hyp_id][1:]
                    completed_hypotheses.append(hyp_tgt_words)  # remove <s> and </s> in completed hypothesis
                    completed_hypothesis_scores.append(new_hyp_score)
                else:
                    if word_id == tgt_vocab.unk_id:
                        if gentoken_new_hyp_unks: word = gentoken_new_hyp_unks[prev_hyp_id]
                        else: word = tgt_vocab.id2word[tgt_vocab.unk_id]
                    else:
                        word = tgt_vocab.id2word[word_id]
                    # import pdb;pdb.set_trace()
                    hyp_tgt_words = hypotheses[prev_hyp_id] + [word]
                    new_hypotheses.append(hyp_tgt_words)
                    new_hypotheses_word_ids.append(hypotheses_word_ids[prev_hyp_id] + [word_id])
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(new_hyp_score)

            if len(completed_hypotheses) == sample_size:
                break

            live_hyp_ids = torch.LongTensor(live_hyp_ids).to(self.device)
            # h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            # att_tm1 = att_t[live_hyp_ids]
            with torch.no_grad():
                hyp_scores = Variable(torch.FloatTensor(new_hyp_scores).to(self.device))  # new_hyp_scores[live_hyp_ids]
            hypotheses = new_hypotheses
            hypotheses_word_ids = new_hypotheses_word_ids

        return completed_hypotheses

        
        


    def sample(self, src_sent, encoder_hidden, src_mask,src_vocab,
                    tgt_vocab,
                    max_len = 100, sample_size = 5, mode=['beam_search', 'sample'][0]):
        """
        Args:
            batch_size = 1
            src_sent: [seq_len]
            encoder_hidden: [batch_size, seq_len, hidden_size]
            src_mask: [batch_size, seq_len]
            src_vocab: Vocab class
            tgt_vocab: Vocab class
            max_len: int
            EOS_idx: int
        """

        assert mode in ['beam_search', 'sample']

        src_token_tgt_vocab_ids = [tgt_vocab[token] for token in src_sent]
        src_unk_pos_list = [pos for pos, token_id in enumerate(src_token_tgt_vocab_ids) if
                            token_id == tgt_vocab.unk_id]

        # sometimes a word may appear multi-times in the source, in this case,
        # we just copy its first appearing position. Therefore we mask the words
        # appearing second and onwards to -1
        token_set = set()
        for i, tid in enumerate(src_token_tgt_vocab_ids):
            if tid in token_set:
                src_token_tgt_vocab_ids[i] = -1
            else:
                token_set.add(tid)
        
        
        t = 0
        eos_id = tgt_vocab['</s>']
        completed_hypotheses = []
        completed_hypothesis_scores = []

        if mode == 'beam_search':
            hypotheses = [['<s>']]
            hypotheses_word_ids = [[tgt_vocab['<s>']]]
        else:
            hypotheses = [['<s>'] for _ in range(sample_size)]
            hypotheses_word_ids = [[tgt_vocab['<s>']] for _ in range(sample_size)]
        with torch.no_grad():
            hyp_scores = Variable(torch.FloatTensor(len(hypotheses)).to(self.device).zero_())

        while len(completed_hypotheses) < sample_size and t < max_len:
            t += 1
            hyp_num = len(hypotheses)
            # encoder_hidden: [batch_size, seq_len, hidden_size]
            expanded_src_encodings = encoder_hidden.expand(hyp_num, encoder_hidden.size(1), encoder_hidden.size(2))
            # src_mask: [batch_size, seq_len]
            expanded_src_mask = src_mask.expand(hyp_num, src_mask.size(1))
            # batch, seq, hidden
            with torch.no_grad():
                already_generated_ys = Variable(torch.LongTensor(hypotheses_word_ids).to(self.device))
            # batch, seq, hidden
            ys_embedded = self.embed(already_generated_ys)

            # batch, hidden
            decoder_output = self.forward_step(ys_embedded, expanded_src_encodings, expanded_src_mask)
            # batch, 2
            tgt_token_predictor = F.softmax(self.tgt_token_predictor(decoder_output), dim=-1)
            # batch, tgt_vocab_size
            token_gen_prob = F.softmax(self.generator(decoder_output), dim=-1)
            
            # ???need to check
            token_copy_prob = self.pointer_network(expanded_src_encodings, expanded_src_mask, decoder_output.unsqueeze(1)).squeeze(1)

            token_gen_prob = tgt_token_predictor[:, 0].unsqueeze(1) * token_gen_prob

            for token_pos, token_vocab_id in enumerate(src_token_tgt_vocab_ids):
                if token_vocab_id != -1 and token_vocab_id != tgt_vocab.unk_id:
                    p_copy = tgt_token_predictor[:, 1] * token_copy_prob[:, token_pos]
                    token_gen_prob[:, token_vocab_id] = token_gen_prob[:, token_vocab_id] + p_copy


            # second, add the probability of copying the most probable unk word
            gentoken_new_hyp_unks = []
            if src_unk_pos_list:
                for hyp_id in range(hyp_num):
                    unk_pos = token_copy_prob[hyp_id][src_unk_pos_list].data.cpu().numpy().argmax()
                    unk_pos = src_unk_pos_list[unk_pos]
                    token = src_sent[unk_pos]
                    gentoken_new_hyp_unks.append(token)

                    unk_copy_score = tgt_token_predictor[hyp_id, 1] * token_copy_prob[hyp_id, unk_pos]
                    token_gen_prob[hyp_id, tgt_vocab.unk_id] = unk_copy_score

            live_hyp_num = sample_size - len(completed_hypotheses)

            if mode == 'beam_search':
                log_token_gen_prob = torch.log(token_gen_prob)
                new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(token_gen_prob) + log_token_gen_prob).view(-1)
                top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=live_hyp_num)
                # import pdb; pdb.set_trace();
                # prev_hyp_ids = (top_new_hyp_pos // len(tgt_vocab)).cpu().data
                prev_hyp_ids = (torch.div(top_new_hyp_pos, len(tgt_vocab), rounding_mode='floor')).cpu().data
                word_ids = (top_new_hyp_pos % len(tgt_vocab)).cpu().data
                top_new_hyp_scores = top_new_hyp_scores.cpu().data
            else:
                word_ids = torch.multinomial(token_gen_prob, num_samples=1)
                prev_hyp_ids = range(live_hyp_num)
                top_new_hyp_scores = hyp_scores + torch.log(torch.gather(token_gen_prob, dim=1, index=word_ids)).squeeze(1)
                top_new_hyp_scores = top_new_hyp_scores.cpu().data
                word_ids = word_ids.view(-1).cpu().data

            new_hypotheses = []
            new_hypotheses_word_ids = []
            live_hyp_ids = []
            new_hyp_scores = []
            for prev_hyp_id, word_id, new_hyp_score in zip(prev_hyp_ids, word_ids, top_new_hyp_scores):
                # import pdb;pdb.set_trace()
                word_id = word_id.item()
                if word_id == eos_id:
                    hyp_tgt_words = hypotheses[prev_hyp_id][1:]
                    completed_hypotheses.append(hyp_tgt_words)  # remove <s> and </s> in completed hypothesis
                    completed_hypothesis_scores.append(new_hyp_score)
                else:
                    if word_id == tgt_vocab.unk_id:
                        if gentoken_new_hyp_unks: word = gentoken_new_hyp_unks[prev_hyp_id]
                        else: word = tgt_vocab.id2word[tgt_vocab.unk_id]
                    else:
                        word = tgt_vocab.id2word[word_id]
                    # import pdb;pdb.set_trace()
                    hyp_tgt_words = hypotheses[prev_hyp_id] + [word]
                    new_hypotheses.append(hyp_tgt_words)
                    new_hypotheses_word_ids.append(hypotheses_word_ids[prev_hyp_id] + [word_id])
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(new_hyp_score)

            if len(completed_hypotheses) == sample_size:
                break

            live_hyp_ids = torch.LongTensor(live_hyp_ids).to(self.device)
            # h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            # att_tm1 = att_t[live_hyp_ids]
            with torch.no_grad():
                hyp_scores = Variable(torch.FloatTensor(new_hyp_scores).to(self.device))  # new_hyp_scores[live_hyp_ids]
            hypotheses = new_hypotheses
            hypotheses_word_ids = new_hypotheses_word_ids

        return completed_hypotheses

def test_EncoderTransformer():
    vocab_size = 10
    embedding_dim = 8 # even number!
    hidden_size = 16
    nlayers = 3
    max_len = 5
    bs = 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    src_input_idx = torch.randint(0, 10, (bs, max_len), dtype=torch.long)
    src_input_idx = src_input_idx.to(device)
    # lengths = torch.tensor([5, 4], dtype=torch.long)
    lengths = [max_len, max_len-1]

    encoder = EncoderTransformer(vocab_size, embedding_dim,
                          hidden_size, nlayers, device)
    encoder.to(device)
    output = encoder(src_input_idx, lengths)

    print(output.shape)
    # batch, seq, hidden


def test_DecoderTransformer():
    action_size = 10
    embedding_dim = 8 # even number!
    encoder_dim = embedding_dim
    hidden_size = 16
    nlayers = 3
    max_len = 5
    bs = 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tgt_action_idx = torch.randint(0, 10, (bs, max_len), dtype=torch.long)
    tgt_action_idx = tgt_action_idx.to(device)
    # lengths = torch.tensor([5, 4], dtype=torch.long)
    lengths = [max_len, max_len-1]

    encoder_output = torch.randn(bs, max_len, encoder_dim).to(device)

    encoder = EncoderTransformer(action_size, embedding_dim,
                                 hidden_size, nlayers, device)
    
    src_mask = encoder.length_array_to_mask_tensor(lengths)


    decoder = DecoderTransformer(
        action_size, encoder_dim, embedding_dim, hidden_size, nlayers, device)
    decoder.to(device)


    output = decoder(tgt_action_idx, encoder_output, src_mask, None)

    print(output.shape)


def test_CopyDecoderTransformer():
    from utils.vocab import VocabEntry
    from utils.nn_utils import to_input_variable

    src_sents = [['start', 'the', 'first', 'sentence', '.'], ['the', 'second', 'sentence', '.']]
    tgt_sents = [['the', 'second', 'sentence', '.'], ['the', 'third', 'sentence', 'end', '.']]


    src_vocab = VocabEntry.from_corpus(src_sents, size=50)
    tgt_vocab = VocabEntry.from_corpus(tgt_sents, size=50)
    print(src_vocab)

    print(src_vocab.to_dict())

    src_inputs = to_input_variable(src_sents, src_vocab, cuda=True).permute(1, 0)
    tgt_inputs = to_input_variable(tgt_sents, tgt_vocab, cuda=True, append_boundary_sym=True).permute(1, 0)
    
    vocab_size = len(src_vocab)
    action_size = len(tgt_vocab)
    embedding_dim = 8 # even number!
    encoder_dim = embedding_dim
    hidden_size = 16
    nlayers = 3
    lengths = [len(e) for e in src_inputs]
    max_len = max(lengths)
    bs = 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    encoder = EncoderTransformer(vocab_size, embedding_dim,
                                 hidden_size, nlayers, device)
    encoder.to(device)
    decoder = CopyDecoderTransformer(action_size, encoder_dim, embedding_dim, hidden_size, nlayers, device)
    decoder.to(device)

    src_mask = encoder.length_array_to_mask_tensor(lengths)

    # train
    encoder.train()
    decoder.train()
    encoder_output = encoder(src_inputs, lengths)
    tgt_token_copy_idx_mask, tgt_token_gen_mask = decoder.get_generate_and_copy_meta_tensor(src_sents,tgt_sents, src_vocab)
    decoder_output = decoder(tgt_inputs, encoder_output, src_mask, None, tgt_token_copy_idx_mask, tgt_token_gen_mask)
    print(decoder_output)

    # eval
    encoder.eval()
    decoder.eval()
    encoder_output = encoder(src_inputs[0].unsqueeze(0), [lengths[0]])
    src_mask = encoder.length_array_to_mask_tensor([lengths[0]])
    decoder_output = decoder.sample(src_sents[0], encoder_output, src_mask, src_vocab, tgt_vocab)
    print(decoder_output)
    
if __name__ == '__main__':
    # test_EncoderTransformer()
    # test_DecoderTransformer()
    # test_CopyDecoderTransformer()
    pass
