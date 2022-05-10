import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from model.copy_transformer import EncoderTransformer, CopyDecoderTransformer
from utils.vocab import VocabEntry
from utils.nn_utils import to_input_variable

class CopyTransformer(torch.nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embedding_dim=256,
                 hidden_size=2048, nlayers=8, use_cuda = True, dropout = 0.2,nhead=8):
        super(CopyTransformer, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_vocab_size = len(src_vocab)
        self.tgt_vocab_size = len(tgt_vocab)

        self.encoder_embedding_dim = embedding_dim
        self.decoder_embedding_dim = embedding_dim
        self.encoder_dim = embedding_dim

        self.encoder_hidden_size = hidden_size
        self.decoder_hidden_size = hidden_size

        self.encoder_nlayers = nlayers
        self.decoder_nlayers = nlayers

        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.encoder_dp = dropout
        self.decoder_dp = dropout

        self.encoder_nhead = nhead
        self.decoder_nhead = nhead

        self.encoder = EncoderTransformer(self.src_vocab_size, self.encoder_embedding_dim,
                                 self.encoder_hidden_size, self.encoder_nlayers, self.device,
                                 self.encoder_dp, self.encoder_nhead)
        '''
        (self, vocab_size, embedding_dim=256, hidden_size=2048, nlayers=6,
                 device=torch.device(
                     'cuda:0' if torch.cuda.is_available() else 'cpu'),
                 dp=0.3,
                 nhead=8):
        '''
        self.decoder = CopyDecoderTransformer(self.tgt_vocab_size, self.encoder_dim, self.decoder_embedding_dim, 
                                            self.decoder_hidden_size, self.decoder_nlayers,
                                            self.device, self.decoder_dp, self.decoder_nhead)
        '''
        (self, action_size, encoder_dim,embedding_dim=256, hidden_size=2048, nlayers=6,
                 device=torch.device(
                     'cuda:0' if torch.cuda.is_available() else 'cpu'),
                 dp=0.3,
                 nhead=8)
        '''
        
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


    def forward(self, src_sents, tgt_sents, src_inputs = None, tgt_inputs = None):
        '''

        '''
        if src_inputs is None:
            src_inputs = to_input_variable(src_sents, self.src_vocab, cuda=self.use_cuda).permute(1, 0)
        if tgt_inputs is None:
            tgt_inputs = to_input_variable(tgt_sents, self.tgt_vocab, cuda=self.use_cuda, append_boundary_sym=True).permute(1, 0)
    
        lengths = [len(e) for e in src_inputs]
        src_mask = self.encoder.length_array_to_mask_tensor(lengths)
        tgt_token_copy_idx_mask, tgt_token_gen_mask = self.decoder.get_generate_and_copy_meta_tensor(src_sents,tgt_sents, self.src_vocab)
        encoder_output = self.encoder(src_inputs, src_mask)
        '''
        (self, input, src_mask, lengths = None):
        '''
        decoder_output = self.decoder(tgt_inputs, encoder_output, src_mask, None, tgt_token_copy_idx_mask, tgt_token_gen_mask)
        '''
        forward(self, targets, encoder_hidden, src_mask, tgt_mask, max_len = None):
        '''
        return decoder_output

    def sample(self, src_sent, src_inputs = None, max_len = 100, sample_size = 5, mode = 'beam_search'):
        
        if src_inputs is None:
            src_inputs = to_input_variable([src_sent], self.src_vocab, cuda=self.use_cuda).permute(1, 0)
        lengths = [len(e) for e in src_inputs]
        src_mask = self.encoder.length_array_to_mask_tensor(lengths)
        encoder_output = self.encoder(src_inputs, src_mask)

        sample_output = self.decoder.sample(src_sent, encoder_output, src_mask, 
                                            self.src_vocab, self.tgt_vocab,
                                            max_len = max_len, sample_size = sample_size, mode = mode)
        '''
        (self, src_sent, encoder_hidden, src_mask,src_vocab,
                    tgt_vocab,
                    max_len = 100, sample_size = 5, mode=['beam_search', 'sample'][0]):
        '''
        return sample_output


def test_CopyTransformer():
    # from utils.vocab import VocabEntry
    # from utils.nn_utils import to_input_variable

    src_sents = [['start', 'the', 'first', 'sentence', '.'], ['the', 'second', 'sentence', '.']]
    tgt_sents = [['the', 'second', 'sentence', '.'], ['the', 'third', 'sentence', 'end', '.']]


    src_vocab = VocabEntry.from_corpus(src_sents, size=50)
    tgt_vocab = VocabEntry.from_corpus(tgt_sents, size=50)
    print(src_vocab)

    src_inputs = to_input_variable(src_sents, src_vocab, cuda=True).permute(1, 0)
    tgt_inputs = to_input_variable(tgt_sents, tgt_vocab, cuda=True, append_boundary_sym=True).permute(1, 0)

    embedding_dim = 8 # even number!
    encoder_dim = embedding_dim
    hidden_size = 16
    nlayers = 3
    lengths = [len(e) for e in src_inputs]
    max_len = max(lengths)
    bs = 2
    model = CopyTransformer(src_vocab, tgt_vocab, embedding_dim, hidden_size, nlayers, use_cuda=True)
    # model.to(torch.device("cuda"))

    model.eval()
    train_out = model.forward(src_sents, tgt_sents, src_inputs, tgt_inputs)
    print(train_out)
    train_out = model(src_sents, tgt_sents)
    print(train_out)

    test_out = model.sample(src_sents[0], src_inputs[0].unsqueeze(0), max_len = max_len, sample_size = 5)
    print(test_out)
    test_out = model.sample(src_sents[1])
    print(test_out)
    
if __name__ == '__main__':
    test_CopyTransformer()
