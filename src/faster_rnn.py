import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import torch.optim as optim
import time
import spacy
import pickle
from torchtext import data
from torchtext import datasets


class Fast_RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        embedded = self.embedding(text)
                
        #embedded = [sent len, batch size, emb dim]
        
        embedded = embedded.permute(1, 0, 2)
        
        #embedded = [batch size, sent len, emb dim]
        
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        
        #pooled = [batch size, embedding_dim]
                
        return self.fc(pooled)

    
    def evaluate_performance(self, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0
        
        self.eval()
        
        with torch.no_grad():
        
            for batch in iterator:

                predictions = self(batch.text).squeeze(1)
                
                loss = criterion(predictions, batch.label)
                
                acc = binary_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def fit(self, train_data, valid_data, test_data):
        BATCH_SIZE = 64

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data), 
            batch_size = BATCH_SIZE, 
            device = device)

        optimizer = optim.Adam(self.parameters())

        criterion = nn.BCEWithLogitsLoss()

        self.to(device)

        criterion = criterion.to(device)
        N_EPOCHS = 5

        best_valid_loss = float('inf')


        for epoch in range(N_EPOCHS):

            start_time = time.time()
            
            train_loss, train_acc = self.fit_epoch(train_iterator, optimizer, criterion)
            valid_loss, valid_acc = self.evaluate_performance(valid_iterator, criterion)
            
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.state_dict(), 'tut2-model.pt')
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


    def fit_epoch(self, iterator, optimizer, criterion):
        
        epoch_loss = 0
        epoch_acc = 0
        
        self.train()
        
        for batch in iterator:
            
            optimizer.zero_grad()
            
            predictions = self(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)
            
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def save_model(self):
        torch.save(self, "faster_rnn_model")


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

    
def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    TEXT = data.Field(tokenize = 'spacy', preprocessing = generate_bigrams)
    LABEL = data.LabelField(dtype = torch.float)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    train_data, valid_data = train_data.split()

    MAX_VOCAB_SIZE = 25_000

    TEXT.build_vocab(train_data, 
                    max_size = MAX_VOCAB_SIZE, 
                    vectors = "glove.6B.100d", 
                    unk_init = torch.Tensor.normal_)

    LABEL.build_vocab(train_data)

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    OUTPUT_DIM = 1
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = Fast_RNN(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX) 

    print(f'The model has {model.count_parameters():,} trainable parameters')

    pretrained_embeddings = TEXT.vocab.vectors

    model.embedding.weight.data.copy_(pretrained_embeddings)    

    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    model.fit(train_data, valid_data, test_data)

    model.save_model()


main()