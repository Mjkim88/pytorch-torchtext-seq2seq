import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils import *
from model import *

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_len, trg_soi):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.trg_soi = trg_soi
        
        self.embed = nn.Embedding(vocab_size, embed_dim)        
        self.attention = Attention(hidden_dim) 
        self.decodercell = DecoderCell(embed_dim, hidden_dim)
        self.dec2word = nn.Linear(hidden_dim, vocab_size)


    def forward(self, enc_h, prev_s, target=None):
        '''
        enc_h  : B x S x 2*H 
        prev_s : B x H
        '''

        if target is not None:
            batch_size, target_len = target.size(0), target.size(1)
            
            dec_h = Variable(torch.zeros(batch_size, target_len, self.hidden_dim))

            if torch.cuda.is_available():
                dec_h = dec_h.cuda()

            target = self.embed(target)  
            for i in range(target_len):
                ctx = self.attention(enc_h, prev_s)                     
                prev_s = self.decodercell(target[:, i], prev_s, ctx)       
                dec_h[:,i,:] = prev_s.unsqueeze(1)

            outputs = self.dec2word(dec_h)

        else:
            batch_size = enc_h.size(0)
            target = Variable(torch.LongTensor([self.trg_soi] * batch_size), volatile=True).view(batch_size, 1)
            outputs = Variable(torch.zeros(batch_size, self.max_len, self.vocab_size))

            if torch.cuda.is_available():
                target = target.cuda()
                outputs = outputs.cuda()
            
            for i in range(self.max_len):
                target = self.embed(target).squeeze(1)              
                ctx = self.attention(enc_h, prev_s)                 
                prev_s = self.decodercell(target, prev_s, ctx)
                output = self.dec2word(prev_s) 
                outputs[:,i,:] = output
                target = output.topk(1)[1]
            
        return outputs


class DecoderCell(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(DecoderCell, self).__init__()

        self.input_weights = nn.Linear(embed_dim, hidden_dim*2)
        self.hidden_weights = nn.Linear(hidden_dim, hidden_dim*2)
        self.ctx_weights = nn.Linear(hidden_dim*2, hidden_dim*2)

        self.input_in = nn.Linear(embed_dim, hidden_dim)
        self.hidden_in = nn.Linear(hidden_dim, hidden_dim)
        self.ctx_in = nn.Linear(hidden_dim*2, hidden_dim)


    def forward(self, trg_word, prev_s, ctx):
        '''
        trg_word : B x E
        prev_s   : B x H 
        ctx      : B x 2*H
        '''
        gates = self.input_weights(trg_word) + self.hidden_weights(prev_s) + self.ctx_weights(ctx)
        reset_gate, update_gate = gates.chunk(2,1)

        reset_gate = F.sigmoid(reset_gate)
        update_gate = F.sigmoid(update_gate)

        prev_s_tilde = self.input_in(trg_word) + self.hidden_in(prev_s) + self.ctx_in(ctx)
        prev_s_tilde = F.tanh(prev_s_tilde)

        prev_s = torch.mul((1-reset_gate), prev_s) + torch.mul(reset_gate, prev_s_tilde)
        return prev_s
        
