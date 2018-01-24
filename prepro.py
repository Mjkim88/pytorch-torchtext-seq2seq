import torch
from torchtext import data
from torchtext import datasets
import time
import re
import spacy
import os 

SOS_WORD = '<SOS>'
EOS_WORD = '<EOS>'
PAD_WORD = '<PAD>'
    

class MaxlenTranslationDataset(data.Dataset):
	# Code modified from 
	# https://github.com/pytorch/text/blob/master/torchtext/datasets/translation.py
	# to be able to control the max length of the source and target sentences

    def __init__(self, path, exts, fields, max_len=None, **kwargs):
	
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.split(' '), trg_line.split(' ')
                if max_len is not None:
                	src_line = src_line[:max_len]
                	src_line = str(' '.join(src_line))
                	trg_line = trg_line[:max_len]
                	trg_line = str(' '.join(trg_line))

                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(MaxlenTranslationDataset, self).__init__(examples, fields, **kwargs)

	
class DataPreprocessor(object):
	def __init__(self):
		self.src_field, self.trg_field = self.generate_fields()

	def preprocess(self, train_path, val_path, src_lang, trg_lang, max_len=None):
		# Generating torchtext dataset class
		train_dataset = self.generate_data(train_path, src_lang, trg_lang, max_len)
		val_dataset = self.generate_data(val_path, src_lang, trg_lang, max_len)

		# Building field vocabulary
		self.src_field.build_vocab(train_dataset, max_size=30000)
		self.trg_field.build_vocab(train_dataset, max_size=30000)

		src_vocab, trg_vocab, src_inv_vocab, trg_inv_vocab = self.generate_vocabs()

		vocabs = {'src_vocab': src_vocab, 'trg_vocab':trg_vocab, 
			  'src_inv_vocab':src_inv_vocab, 'trg_inv_vocab':trg_inv_vocab}

		return train_dataset, val_dataset, vocabs

	def load_data(self, data_file):

		# Loading saved data
		dataset = torch.load(data_file)
		train_examples = dataset['train_examples']
		val_examples = dataset['val_examples']

		# Generating torchtext dataset class
		fields = [('src', self.src_field), ('trg', self.trg_field)]
		train_dataset = data.Dataset(fields=fields, examples=train_examples)	
		val_dataset = data.Dataset(fields=fields, examples=val_examples)	

		# Building field vocabulary
		self.src_field.build_vocab(train_dataset, max_size=30000)
		self.trg_field.build_vocab(train_dataset, max_size=30000)

		src_vocab, trg_vocab, src_inv_vocab, trg_inv_vocab = self.generate_vocabs()
		vocabs = {'src_vocab': src_vocab, 'trg_vocab':trg_vocab, 
			  'src_inv_vocab':src_inv_vocab, 'trg_inv_vocab':trg_inv_vocab}
		
		return train_dataset, val_dataset, vocabs	


	def save_data(self, data_file, train_dataset, val_dataset):

		train_examples = vars(train_dataset)['examples']
		train_fields = vars(train_dataset)['fields']
		val_examples = vars(val_dataset)['examples']
		val_fields = vars(val_dataset)['fields']		

		dataset = {'train_examples': train_examples, 'val_examples': val_examples}
		
		torch.save(dataset, data_file)

	def generate_fields(self):     
	    src_field = data.Field(tokenize=data.get_tokenizer('spacy'), 
	                           init_token=SOS_WORD,
	                           eos_token=EOS_WORD,
	                           pad_token=PAD_WORD,
	                           include_lengths=True,
	                           batch_first=True)
	    
	    trg_field = data.Field(tokenize=data.get_tokenizer('spacy'), 
	                           init_token=SOS_WORD,
	                           eos_token=EOS_WORD,
	                           pad_token=PAD_WORD,
	                           include_lengths=True,
	                           batch_first=True)

	    return src_field, trg_field

	def generate_data(self, data_path, src_lang, trg_lang, max_len=None):
	    exts = ('.'+src_lang, '.'+trg_lang)

	    dataset = MaxlenTranslationDataset(
	        path=data_path,
	        exts=(exts),
	        fields=(self.src_field, self.trg_field),
	        max_len=max_len)    

	    return dataset

	def generate_vocabs(self):
	    # Define string to index vocabs
	    src_vocab = self.src_field.vocab.stoi
	    trg_vocab = self.trg_field.vocab.stoi
	
	    # Define index to string vocabs
	    src_inv_vocab = self.src_field.vocab.itos
	    trg_inv_vocab = self.trg_field.vocab.itos

	    return src_vocab, trg_vocab, src_inv_vocab, trg_inv_vocab



