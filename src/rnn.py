import torch.nn as nn
import torch
import random
import torch.optim as optim
import time
import spacy
from torchtext import data
from torchtext import datasets


class RNN(nn.Module):
    def __init__(self, text, embedding_dim=100, hidden_dim=256, output_dim=1, n_layers=4, 
                 bidirectional=True, dropout=0.5):

        super().__init__()

        self.text = text

        self.nlp = spacy.load('en')

        self.pad_idx = text.vocab.stoi[text.pad_token]
        self.input_dim = len(text.vocab)
        self.embedding_dim = embedding_dim
        
        self.embedding = nn.Embedding(self.input_dim, embedding_dim, padding_idx = self.pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
                
        #hidden = [batch size, hid dim * num directions]
            
        return self.fc(hidden)

    
    def fit_model(self, train_data, test_data, labels):
        train_data, valid_data = train_data.split()
        
        labels.build_vocab(train_data)

        BATCH_SIZE = 64

        # use gpu for calculations if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data), 
            batch_size = BATCH_SIZE,
            sort_within_batch = True,
            device = device)

        print(f'The model has {self.count_parameters():,} trainable parameters')

        pretrained_embeddings = self.text.vocab.vectors

        print(pretrained_embeddings.shape)

        self.embedding.weight.data.copy_(pretrained_embeddings)

        UNK_IDX = self.text.vocab.stoi[self.text.unk_token]

        self.embedding.weight.data[UNK_IDX] = torch.zeros(self.embedding_dim)
        self.embedding.weight.data[self.pad_idx] = torch.zeros(self.embedding_dim)

        print(self.embedding.weight.data)


        optimizer = optim.Adam(self.parameters())
        criterion = nn.BCEWithLogitsLoss()

        self = self.to(device)
        criterion = criterion.to(device)

        N_EPOCHS = 10

        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):

            start_time = time.time()
            
            train_loss, train_acc = self.fit_epoch(train_iterator, optimizer, criterion)
            valid_loss, valid_acc = self.evaluate(valid_iterator, criterion)
            
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.state_dict(), 'tut2-model.pt')
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        self.load_state_dict(torch.load('tut2-model.pt'))

        test_loss, test_acc = self.evaluate(test_iterator, criterion)

        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def accuracy(self, preds, y):    
        #round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float() #convert into float for division 
        acc = correct.sum() / len(correct)
        return acc


    def fit_epoch(self, iterator, optimizer, criterion):
        
        epoch_loss = 0
        epoch_acc = 0
        
        self.train()
        
        for batch in iterator:
            
            optimizer.zero_grad()
            
            text, text_lengths = batch.text
            
            predictions = self(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = self.accuracy(predictions, batch.label)
            
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)


    def evaluate(self, iterator, criterion):
        
        epoch_loss = 0
        epoch_acc = 0
        
        self.eval()
        
        with torch.no_grad():
        
            for batch in iterator:

                text, text_lengths = batch.text
                
                predictions = self(text, text_lengths).squeeze(1)
                
                loss = criterion(predictions, batch.label)
                
                acc = self.accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)


    def predict_sentiment(self, sentence, device, text):
        self.eval()
        
        tokenized = [tok.text for tok in self.nlp.tokenizer(sentence)]
        indexed = [text.vocab.stoi[t] for t in tokenized]
        length = [len(indexed)]
        
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        length_tensor = torch.LongTensor(length)
        
        prediction = torch.sigmoid(self(tensor, length_tensor))
        
        print(prediction.item())
        
        return prediction.item()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs