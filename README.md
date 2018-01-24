# Pytorch-Torchtext-Seq2Seq
This is a [Pytorch](https://github.com/pytorch/pytorch)
implementation of [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)


## Prerequisites
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.2.0](http://pytorch.org/)
* [Torchtext 0.2.1](https://github.com/pytorch/text)
* [spaCy 2.0.5](https://spacy.io/)
* [TensorFlow 1.3+](https://www.tensorflow.org/) (optional for tensorboard)


## Getting Started
### 1. Clone the repository
```bash
$ git clone https://github.com/Mjkim88/Pytorch-Torchtext-Seq2Seq.git
$ cd Pytorch-Torchtext-Seq2Seq
```

### 2. Download the dataset
```bash
$ bash download.sh
```
This commands will download Europarl v7(for training) and News Commentary(for validation) datasets to `data/` folder. If you want to train and test other datasets, you don't need to run this command. 

### 3. Train the model 
```bash
$ python main.py --dataset 'europarl_news' --src_lang 'de' --trg_lang 'en' --data_path './data/' \
                 --train_path './data/europarl/europarl-v7.de-en' --val_path './data/news/news-commentary-v9.de-en' \
                 --log log --sample sample
```
If you initially run the above command, the model starts from preprocessing data using Torchtext and automatically saves the results, 
so that you don't need to preprocess the data again. Then it trains and test 

### (Optional) Tensorboard visualization 
```bash
$ tensorboard --logdir='./logs/' --
```
For the tensorboard visualization, open the new terminal and run the command below and open `http://localhost:8888` on your web browser.
