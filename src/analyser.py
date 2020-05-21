import torch
import torch.nn as nn
import random
import torch.optim as optim
import time
import spacy

from torchtext import data
from torchtext import datasets
from rnn import RNN

nlp = spacy.load('en')

def main():
    text = data.Field(tokenize = 'spacy', include_lengths = True)
    labels = data.LabelField(dtype = torch.float32)

    # split IMDb data into train, test sets
    train_data, test_data = datasets.IMDB.splits(text, labels)

    MAX_VOCAB_SIZE = 25000

    text.build_vocab(train_data, 
        max_size = MAX_VOCAB_SIZE, 
        vectors = "glove.6B.100d", 
        unk_init = torch.Tensor.normal_)

    model = RNN(text)

    model.fit_model(train_data, test_data, labels)

main()