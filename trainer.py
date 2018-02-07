import torch
from torch import cuda
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *

import numpy as np
import math
import time
import os

from logger import Logger
from tqdm import tqdm

from prepro import *
from utils import *
from model.Seq2Seq import Seq2Seq
from bleu import *


class Trainer(object):
    def __init__(self, train_loader, val_loader, vocabs, args):

        # Language setting
        self.max_len = args.max_len

        # Data Loader
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Path
        self.data_path = args.data_path
        self.sample_path = os.path.join('./samples/' + args.sample)
        self.log_path = os.path.join('./logs/' + args.log) 

        if not os.path.exists(self.sample_path): os.makedirs(self.sample_path)
        if not os.path.exists(self.log_path): os.makedirs(self.log_path)

        # Hyper-parameters
        self.lr = args.lr
        self.grad_clip = args.grad_clip
        self.embed_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim
        self.num_layer = args.num_layer

        # Training setting
        self.batch_size = args.batch_size
        self.num_epoch = args.num_epoch
        self.iter_per_epoch = len(train_loader)

        # Log
        self.logger = open(self.log_path+'/log.txt','w')
        self.sample = open(self.sample_path+'/sample.txt','w')
        self.tf_log = Logger(self.log_path)

        self.build_model(vocabs)


    def build_model(self, vocabs):
        # build dictionaries
        self.src_vocab = vocabs['src_vocab']
        self.trg_vocab = vocabs['trg_vocab']
        self.src_inv_vocab = vocabs['src_inv_vocab']
        self.trg_inv_vocab = vocabs['trg_inv_vocab']
        self.trg_soi = self.trg_vocab[SOS_WORD]

        self.src_nword = len(self.src_vocab)
        self.trg_nword = len(self.trg_vocab)
        
        # build the model
        self.model = Seq2Seq(self.src_nword, self.trg_nword, self.num_layer, self.embed_dim, self.hidden_dim, 
        					  self.max_len, self.trg_soi)

        # set the criterion and optimizer
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.8)
        
        if torch.cuda.is_available():
            self.model.cuda()

        print (self.model)
        print (self.criterion)
        print (self.optimizer)



    def train(self):
        self.best_bleu = .0
        
        for epoch in range(self.num_epoch):
            #self.scheduler.step()
            self.train_loss = AverageMeter()
            self.train_bleu = AverageMeter()
            start_time = time.time()

            for i, batch in enumerate(tqdm(self.train_loader)):
                self.model.train()

                src_input = batch.src[0]; src_length = batch.src[1]
                trg_input = batch.trg[0][:,:-1]; trg_output=batch.trg[0][:,1:]; trg_length = batch.trg[1]
                batch_size, trg_len = trg_input.size(0), trg_input.size(1)

                decoder_logit = self.model(src_input, src_length.tolist(), trg_input)
                pred = decoder_logit.view(batch_size, trg_len, -1)

                self.optimizer.zero_grad()
                loss = self.criterion(decoder_logit, trg_output.contiguous().view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                # Compute BLEU score and Loss
                pred_sents = []
                trg_sents = []
                for j in range(batch_size):
                    pred_sent = self.get_sentence(tensor2np(pred[j]).argmax(axis=-1), 'trg')
                    trg_sent = self.get_sentence(tensor2np(trg_output[j]), 'trg')
                    pred_sents.append(pred_sent)
                    trg_sents.append(trg_sent)
                bleu_value = get_bleu(pred_sents, trg_sents)
                self.train_bleu.update(bleu_value, 1)
                self.train_loss.update(loss.data[0], batch_size)

                if i % 5000 == 0 and i != 0:
                    self.print_train_result(epoch, i, start_time)
                    self.print_sample(batch_size, epoch, i, src_input, trg_output, pred)
                    self.eval(epoch, i)
                    self.train_loss = AverageMeter()
                    self.train_bleu = AverageMeter()
                    start_time = time.time()

                    # Logging tensorboard
                    info = {
                        'epoch': epoch,
                        'train_iter': i,
                        'train_loss': self.train_loss.avg,
                        'train_bleu': self.train_bleu.avg               
                        }
                    for tag, value in info.items():
                        self.tf_log.scalar_summary(tag, value, (epoch * self.iter_per_epoch)+i+1)

            self.print_train_result(epoch, i, start_time)
            self.print_sample(batch_size, epoch, i, src_input, trg_output, pred)           
            self.eval(epoch, i)                


    def eval(self, epoch, train_iter):
        self.model.eval()
        val_bleu = AverageMeter()
        start_time = time.time()
        
        for i, batch in enumerate(tqdm(self.val_loader)):
            src_input = batch.src[0]; src_length = batch.src[1]
            trg_input = batch.trg[0][:,:-1]; trg_output=batch.trg[0][:,1:]; trg_length = batch.trg[1]
            batch_size, trg_len = trg_input.size(0), trg_input.size(1)

            decoder_logit = self.model(src_input, src_length.tolist())
            pred = decoder_logit.view(batch_size, self.max_len, -1)

            # Compute BLEU score
            pred_sents = []
            trg_sents = []            
            for j in range(batch_size):
                pred_sent = self.get_sentence(tensor2np(pred[j]).argmax(axis=-1), 'trg')
                trg_sent = self.get_sentence(tensor2np(trg_output[j]), 'trg')
                pred_sents.append(pred_sent)
                trg_sents.append(trg_sent)
            bleu_value = get_bleu(pred_sents, trg_sents)
            val_bleu.update(bleu_value, 1)
               
        self.print_valid_result(epoch, train_iter, val_bleu.avg, start_time)       
        self.print_sample(batch_size, epoch, train_iter, src_input, trg_output, pred)

        # Save model if bleu score is higher than the best 
        if self.best_bleu < val_bleu.avg:
            self.best_bleu = val_bleu.avg        
            checkpoint = {
                    'model': self.model,
                    'epoch': epoch
                }
            torch.save(checkpoint, self.log_path+'/Model_e%d_i%d_%.3f.pt' % (epoch, train_iter, val_bleu.avg))                 

        # Logging tensorboard
        info = {
            'epoch': epoch,
            'train_iter': train_iter,
            'train_loss': self.train_loss.avg,
            'train_bleu': self.train_bleu.avg,
            'bleu': val_bleu.avg                
            }

        for tag, value in info.items():
            self.tf_log.scalar_summary(tag, value, (epoch * self.iter_per_epoch)+train_iter+1)


    def get_sentence(self, sentence, side):
        def _eos_parsing(sentence):
            if EOS_WORD in sentence:
                return sentence[:sentence.index(EOS_WORD)+1]
            else:
                return sentence

        # index sentence to word sentence                      
        if side == 'trg':
            sentence = [self.trg_inv_vocab[x] for x in sentence]
        else:
            sentence = [self.src_inv_vocab[x] for x in sentence]

        return _eos_parsing(sentence)


    def print_train_result(self, epoch, train_iter, start_time):
        mode = ("=================================        Train         ====================================")
        print (mode, '\n')
        self.logger.write(mode+'\n')
        self.sample.write(mode+'\n')
        
        message = "Train epoch: %d  iter: %d  train loss: %1.3f  train bleu: %1.3f  elapsed: %1.3f " % (
        epoch, train_iter, self.train_loss.avg, self.train_bleu.avg, time.time() - start_time)
        print (message, '\n\n')
        self.logger.write(message+'\n\n')   


    def print_valid_result(self, epoch, train_iter, val_bleu, start_time):
        mode = ("=================================        Validation         ====================================")
        print (mode, '\n')
        self.logger.write(mode+'\n')
        self.sample.write(mode+'\n')

        message = "Train epoch: %d  iter: %d  train loss: %1.3f  train_bleu:  %1.3f  val bleu score: %1.3f  elapsed: %1.3f " % (
        epoch, train_iter, self.train_loss.avg, self.train_bleu.avg, val_bleu, time.time() - start_time)
        print (message, '\n\n' )
        self.logger.write(message+'\n\n')
        

    def print_sample(self, batch_size, epoch, train_iter, source, target, pred):

        def _write_and_print(message):
            for x in message:
                self.sample.write(x+'\n')
            print ((" ").join(message))

        random_idx = randomChoice(batch_size)
        src_sample = self.get_sentence(tensor2np(source)[random_idx], 'src')
        trg_sample = self.get_sentence(tensor2np(target)[random_idx], 'trg')
        pred_sample = self.get_sentence(tensor2np(pred[random_idx]).argmax(axis=-1), 'trg')

        src_message = ["Source Sentence:    ", (" ").join(src_sample), '\n']
        trg_message = ["Target Sentence:    ", (" ").join(trg_sample), '\n']
        pred_message =  ["Generated Sentence: ", (" ").join(pred_sample), '\n']

        message = "Train epoch: %d  iter: %d " % (epoch, train_iter)
        self.sample.write(message+'\n')
        _write_and_print(src_message)
        _write_and_print(trg_message)
        _write_and_print(pred_message)
        self.sample.write('\n\n\n')
