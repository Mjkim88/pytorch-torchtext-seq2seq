# Pytorch-Torchtext-Seq2Seq
[Pytorch](https://github.com/pytorch/pytorch)
implementation of [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473).


### Prerequisites
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.2.0](http://pytorch.org/)
* [Torchtext 0.2.1](https://github.com/pytorch/text)
* [spaCy 2.0.5](https://spacy.io/)
* [TensorFlow 1.3+](https://www.tensorflow.org/) (optional for tensorboard)


## Getting Started
#### 1. Clone the repository
```bash
$ git clone https://github.com/Mjkim88/Pytorch-Torchtext-Seq2Seq.git
$ cd Pytorch-Torchtext-Seq2Seq
```

#### 2. Download the dataset
```bash
$ bash download.sh
```
This commands will download Europarl v7 and dev datasets to `data/` folder. 
If you want to use other datasets, you don't need to run this command. 

#### 3. Train the model 
```bash
$ python main.py --dataset 'europarl' --src_lang 'fr' --trg_lang 'en' --data_path './data' \
                 --train_path './data/training/europarl-v7.fr-en' --val_path './data/dev/newstest2013' \
                 --log log --sample sample
```
If you initially run the above command, the model starts from preprocessing data using Torchtext and automatically saves the preprocessed JSON file to `/data`, so that it avoids preprocessing the same datasets again. 

#### (Optional) Tensorboard visualization 
```bash
$ tensorboard --logdir='./logs' --port=8888
```
For the tensorboard visualization, open the new terminal and run the command below and open `http://localhost:8888` on your web browser.
