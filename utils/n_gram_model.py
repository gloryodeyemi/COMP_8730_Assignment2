import dill as pickle
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import word_tokenize 
import os.path

# function to tokenize our corpus
def tokenize_corpus(brown_corpus):
    tokenized_sentences = []
    for sentence in brown_corpus:
        sentence = ' '.join(sentence)
        tokenized_sentences.append(word_tokenize(sentence.lower()))
    return tokenized_sentences

# functions to train, save, and load the model
def train_model(n, tokenized_corpus):
    train_data, padded_sentences = padded_everygram_pipeline(n, tokenized_corpus)
    model = MLE(n)
    model.fit(train_data, padded_sentences)
    return model

def save_model(n, model):
    pickle.dump(model, open(os.path.join('models', f'{n}_ngram_model.pickle'), 'wb'))
    
def load_model(n):
    loaded_model = pickle.load(open(os.path.join('models', f'{n}_ngram_model.pickle'), 'rb'))
    return loaded_model

