import torch
import spacy
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchtext import data, datasets
from faster_rnn import Fast_RNN, generate_bigrams

app = Flask(__name__)
CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'

nlp = spacy.load('en')
model = torch.load("faster_rnn_model")


TEXT = data.Field(tokenize = 'spacy', preprocessing = generate_bigrams)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_VOCAB_SIZE = 25_000
LABEL = data.LabelField(dtype = torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
TEXT.build_vocab(train_data, 
    max_size = MAX_VOCAB_SIZE, 
    vectors = "glove.6B.100d", 
    unk_init = torch.Tensor.normal_)

@app.route('/analyseText', methods=['POST'])
def analyseText():
    try:
        request_data = request.get_json()

        text = request_data['text']
        detailed = request_data['detailed']

    except Exception as e:
        print(e)
        return jsonify({
            'error': 'Error accessing request data',
            'analysis': []
        })
    
    finally: 

        paragraphs = text.split('\n')

        average_paragraph_sentiments = []

        for p in paragraphs: 
            sentences = p.split(' ')
            sentiment = 0
            for s in sentences:
                sentiment = sentiment + predict_sentiment(model, s)
            
            average_paragraph_sentiment = -1

            if (len(sentences) > 0):
                average_paragraph_sentiment = sentiment / len(sentences)

            average_paragraph_sentiments.append(average_paragraph_sentiment)

        return jsonify({
                'paragraph_sentiments': average_paragraph_sentiments
        })


def predict_sentiment(model, sentence):
    if (len(sentence) == 0):
        return -1
    
    model.eval()
    tokenized = generate_bigrams([tok.text for tok in nlp.tokenizer(sentence)])
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()