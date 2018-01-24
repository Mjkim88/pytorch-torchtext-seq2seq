import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from utils import *
from model import *

class Seq2Seq(nn.Module):
    def __init__(self, src_nword, trg_nword, num_layer, embed_dim, hidden_dim, max_len, trg_soi):
        super(Seq2Seq, self).__init__()

        self.hidden_dim = hidden_dim
        self.trg_nword = trg_nword

        self.encoder = Encoder(src_nword, embed_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = Decoder(trg_nword, embed_dim, hidden_dim, max_len, trg_soi)

    
    def forward(self, source, src_length=None, target=None):
        batch_size = source.size(0)
        
        enc_h, enc_h_t = self.encoder(source, src_length) # B x S x 2*H / 2 x B x H 
        
        dec_h0 = enc_h_t[-1] # B x H 
        dec_h0 = F.tanh(self.linear(dec_h0)) # B x 1 x 2*H

        out = self.decoder(enc_h, dec_h0, target) # B x S x H
        out = F.log_softmax(out.contiguous().view(-1, self.trg_nword))

        return out
