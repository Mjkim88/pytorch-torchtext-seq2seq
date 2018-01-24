import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 

from utils import *


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        self.enc_h_in = nn.Linear(hidden_dim*2, hidden_dim)
        self.prev_s_in = nn.Linear(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, enc_h, prev_s):
        '''
        enc_h  : B x S x 2*H 
        prev_s : B x 1 x H 
        '''
        seq_len = enc_h.size(1) 

        enc_h_in = self.enc_h_in(enc_h) # B x S x H
        prev_s = self.prev_s_in(prev_s).unsqueeze(1)  # B x 1 x H

        h = F.tanh(enc_h_in + prev_s.expand_as(enc_h_in)) # B x S x H
        h = self.linear(h)  # B x S x 1

        alpha = F.softmax(h)
        ctx = torch.bmm(alpha.transpose(2,1), enc_h).squeeze(1) # B x 1 x 2*H

        return ctx  
