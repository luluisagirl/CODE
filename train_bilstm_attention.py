import codecs
import csv
import random

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import re

import multiprocessing
from nltk.corpus import stopwords

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

import keras
from keras.preprocessing import sequence
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.engine.topology import Layer
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping


cpu_count = multiprocessing.cpu_count()  # 4
vocab_dim = 100
ATT_SIZE = 50
n_iterations = 1
n_exposures = 10 
window_size = 7
n_epoch = 30
maxlen = 100
batch_size = 64


def tokenizer(texts):
    my_Token = []
    for item in texts:
        item_text = BeautifulSoup(item , 'html.parser').get_text()  

        item_text = re.sub("[^a-zA-Z]", " ", item_text)  

        words = item_text.lower().split()  

        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops] 

        my_Token.append(words)
        return my_Token


def parse_data(review_data, index):
    data = []
    for sentence in review_data:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(indx[word])
            except:
                new_txt.append(0)  
        data.append(new_txt)
    data = sequence.pad_sequences(data, maxlen=maxlen)  
    return data





class AttentionLayer(Layer):
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size

        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                 initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        self.V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]



def train_bilstm_att(n_symbols, embedding_weights, x_train, y_train, ATT_SIZE):
    print('Defining a Simple Keras Model...')
    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        weights=[embedding_weights],
                        input_length=maxlen))

    model.add(Bidirectional(LSTM(output_dim=50, dropout=0.5, return_sequences=True)))
    model.add(AttentionLayer(attention_size=ATT_SIZE))
    model.add(Dense(2, activation='softmax')) 
    print('Compiling the Model...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    print("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch)

    model.save('model/bilstmAtt_100_05_att50.h5')


if __name__ == '__main__':
 
  
    train_data = pd.read_csv('word2vec-nlp-tutorial/labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
    test_data = pd.read_csv('word2vec-nlp-tutorial/testData.tsv', header=0, delimiter='\t', quoting=3)
    unlabeled = pd.read_csv('word2vec-nlp-tutorial/unlabeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
    date_set = np.concatenate((train_data['review'], test_data['review'], unlabeled['review']))

    print(len(date_set))
 
    date_set = tokenizer(date_set)
    
    model=word2vec
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(),
                        allow_update=True)
    w2indx = {v: k + 1 for k, v in gensim_dict.items()} 
    f = open("word2index.txt", 'w', encoding='utf8')
    for key in w2indx:
        f.write(str(key))
        f.write(' ')
        f.write(str(w2indx[key]))
        f.write('\n')
    f.close()
    w2vec = {word: model[word] for word in w2indx.keys()}  

    index_dict=w2indx 
    word_vectors =w2vec
    n_symbols = len(index_dict) + 1 
    embedding_weights = np.zeros((n_symbols, vocab_dim)) 
    for word, index in index_dict.items(): 
        embedding_weights[index, :] = word_vectors[word]

    neg_train = pd.read_csv('data/neg_train.csv', header=None, index_col=None)
    pos_train = pd.read_csv('data/pos_train.csv', header=None, index_col=None)

    x_train = np.concatenate((neg_train[0], pos_train[0]))
    x_train = tokenizer(x_train)
    x_train = parse_dataset(x_train, index_dict)
    y_train = np.concatenate((np.zeros(len(neg_train), dtype=int), np.ones(len(pos_train), dtype=int)))
    y_train = keras.utils.to_categorical(y_train, num_classes=2)  

    x_test = pd.read_csv('data/test_data.csv', header=None, index_col=None)
    x_test = tokenizer(x_test[0])
    x_test = parse_dataset(x_test, index_dict)

    print(x_train.shape, y_train.shape)
    train_bilstm_att(n_symbols, embedding_weights, x_train, y_train, ATT_SIZE)
    print('load bilstm_model...')
    model = load_model('model/bilstmAtt_100_05_att50.h5', custom_objects={'AttentionLayer':AttentionLayer})
    y_pred = model.predict(x_test)

    for i in range(len(y_pred)):
        max_value = max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value == y_pred[i][j]:
                y_pred[i][j] = 1
            else:
                y_pred[i][j] = 0
    test_result = []
    for i in y_pred:
        if i[0] == 1:
            test_result.append(0)
        else:
            test_result.append(1)

    id = pd.read_csv('word2vec-nlp-tutorial/sampleSubmission.csv', header=0)['id']
    print(len(id))
    print(len(test_result))
    f = codecs.open('data/Submission_bilstmAtt_100_05_att50.csv', 'w', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow(['id', 'sentiment'])
    for i in range(len(id)):
        writer.writerow([id[i],test_result[i]])
    f.close()

