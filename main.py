import argparse
import os
import time
from torch.backends import cudnn

from prepro import *
from trainer import *


def main(args):
	cuda.set_device(int(args.gpu_num))
	cudnn.benchmark = True

	# Load dataset
	train_file = os.path.join(args.data_path, "data_{}_{}_{}_{}.json".format(args.dataset, args.src_lang, 
										args.trg_lang, args.max_len))
	val_file = os.path.join(args.data_path, "data_dev_{}_{}_{}.json".format(args.src_lang, args.trg_lang, args.max_len))

	start_time = time.time() 
	if os.path.isfile(train_file) and os.path.isfile(val_file):
		print ("Loading data..")
		dp = DataPreprocessor()
		train_dataset, val_dataset, vocabs = dp.load_data(train_file, val_file)
	else:	
		print ("Preprocessing data..")		   
		dp = DataPreprocessor()
		train_dataset, val_dataset, vocabs = dp.preprocess(args.train_path, args.val_path, train_file, val_file,
								   args.src_lang, args.trg_lang, args.max_len)
		
		
	print ("Elapsed Time: %1.3f \n"  %(time.time() - start_time))

	print ("=========== Data Stat ===========")
	print ("Train: ", len(train_dataset))
	print ("val: ", len(val_dataset))
	print ("=================================")

	train_loader = data.BucketIterator(dataset=train_dataset, batch_size=args.batch_size, 
	                                   repeat=False, shuffle=True, sort_within_batch=True,
	                                   sort_key=lambda x: len(x.src))
	val_loader = data.BucketIterator(dataset=val_dataset, batch_size=args.batch_size, 
	                                 repeat=False, shuffle=True, sort_within_batch=True,
	                                 sort_key=lambda x: len(x.src)) 

	trainer = Trainer(train_loader, val_loader, vocabs, args)
	trainer.train()


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	# Language setting
	parser.add_argument('--dataset', type=str, default='europarl')
	parser.add_argument('--src_lang', type=str, default='fr')
	parser.add_argument('--trg_lang', type=str, default='en')
	parser.add_argument('--max_len', type=int, default=50)

	# Model hyper-parameters
	parser.add_argument('--lr', type=float, default=0.0001)	
	parser.add_argument('--grad_clip', type=float, default=2)	
	parser.add_argument('--num_layer', type=int, default=2)
	parser.add_argument('--embed_dim',  type=int, default=512)
	parser.add_argument('--hidden_dim', type=int, default=1024)

	# Training setting
	parser.add_argument('--batch_size', type=int, default=40)
	parser.add_argument('--num_epoch', type=int, default=100)	

	# Path
	parser.add_argument('--data_path', type=str, default='./data/')
	parser.add_argument('--train_path', type=str, default='./data/training/europarl-v7.fr-en')
	parser.add_argument('--val_path', type=str, default='./data/dev/newstest2013')

	# Dir. 
	parser.add_argument('--log', type=str, default='log')
	parser.add_argument('--sample', type=str, default='sample')
	
	# Misc.
	parser.add_argument('--gpu_num', type=int, default=0)

	args = parser.parse_args()
	print (args)
	main(args)
