#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: Linlifang
# @file: utils.py
# @time: 18-8-30下午3:50


import pandas as pd
import numpy as np

from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# load data
df = pd.read_csv("yelp_2013.csv")
# df = df.sample(5000)

Y = df.stars.values - 1
Y = to_categorical(Y, num_classes=5)
X = df.text.values

# set hyper parameters
MAX_NUM_WORDS = 30000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
NUM_FILTERS = 50
MAX_LEN = 512
Batch_size = 100
EPOCHS = 10

# shuffle the data
indices = np.arange(X.shape[0])
np.random.seed(2018)
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

# training set, validation set and testing set
nb_validation_samples_val = int((VALIDATION_SPLIT + TEST_SPLIT) * X.shape[0])
nb_validation_samples_test = int(TEST_SPLIT * X.shape[0])

x_train = X[:-nb_validation_samples_val]
y_train = Y[:-nb_validation_samples_val]
x_val = X[-nb_validation_samples_val:-nb_validation_samples_test]
y_val = Y[-nb_validation_samples_val:-nb_validation_samples_test]
x_test = X[-nb_validation_samples_test:]
y_test = Y[-nb_validation_samples_test:]

# use tokenizer to build vocab
tokenizer1 = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer1.fit_on_texts(df.text)
vocab = tokenizer1.word_index

x_train_word_ids = tokenizer1.texts_to_sequences(x_train)
x_test_word_ids = tokenizer1.texts_to_sequences(x_test)
x_val_word_ids = tokenizer1.texts_to_sequences(x_val)

# pad sequences into the same length
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=MAX_LEN)
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=MAX_LEN)
x_val_padded_seqs = pad_sequences(x_val_word_ids, maxlen=MAX_LEN)

# load pre-trained GloVe word embeddings
print("Using GloVe embeddings")
glove_path = 'glove.6B.200d.txt'
embeddings_index = {}
f = open(glove_path)
for line in f:
	values = line.strip().split()
	word = values[0]
	coefs = np.array(values[1:], dtype=np.float32)
	embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

# use pre-trained GloVe word embeddings to initialize the embedding layer
embedding_matrix = np.empty((MAX_NUM_WORDS + 1, EMBEDDING_DIM), dtype=np.float32)
for word, i in vocab.items():
	if i < MAX_NUM_WORDS:
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be random initialized.
			embedding_matrix[i] = embedding_vector
