import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import *


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2): 
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True, )
        
    def forward(self, source, src_length=None, hidden=None):
        '''
        source: B x T 
        '''
        batch_size = source.size(0)
        src_embed = self.embedding(source)
        
        if hidden is None:
            h_size = (self.num_layers *2, batch_size, self.hidden_dim)
            enc_h_0 = Variable(src_embed.data.new(*h_size).zero_(), requires_grad=False)

        if src_length is not None:
            src_embed = nn.utils.rnn.pack_padded_sequence(src_embed, src_length, batch_first=True)

        enc_h, enc_h_t = self.gru(src_embed, enc_h_0) 

        if src_length is not None:
            enc_h, _ = nn.utils.rnn.pad_packed_sequence(enc_h, batch_first=True)

        return enc_h, enc_h_t
