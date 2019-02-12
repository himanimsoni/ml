import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as pyplot
from sklearn.manifold import TSNE
from keras import metrics
from sklearn import metrics
import numpy as np
import pandas as pd
import os
import pickle
import itertools
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, Flatten, Dense, Dropout
from keras.layers import ThresholdedReLU, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.layers import Input
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import Model
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras import backend as K
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
import random
random.seed(7)
# coding=utf-8
# -*- encoding: utf-8 -*-


def preprocess_data(dataset, x_column, y_column):
    x = sequence.pad_sequences(
        dataset[x_column].map(lambda s: [ord(d) for d in s.encode('utf-8').decode('unicode_escape').encode('ascii','ignore').decode("utf-8")]),maxlen=327)
    y = dataset[y_column].values
    return x, y

train = pd.read_csv('en_training_csv.csv',engine='python')
X,y = preprocess_data(train,"text","HS")
#print(X[1498])


#[4] lstm tweets models - 0.707
main_input = Input(shape=(327, ), name='main_input')
embedding = Embedding(128, 128, input_length=327)(main_input)
bi_lstm = Bidirectional(layer=LSTM(64, return_sequences=False,recurrent_dropout=0.5), merge_mode='concat')(embedding)
#x = Dropout(0.5)(bi_lstm)
output = Dense(1, activation='sigmoid')(bi_lstm)
model = Model(inputs=main_input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='best_model_tweet4.h5', monitor='val_loss', save_best_only=True)]
print(model.summary())

'''
#nyu model
main_input = Input(shape=(327, ), name='main_input')
embedding = Embedding(input_dim=128, output_dim=128,input_length=327)(main_input)
lstm = LSTM(128, return_sequences=False)(embedding)
drop = Dropout(0.5)(lstm)
output = Dense(1, activation='sigmoid')(drop)
model = Model(inputs=main_input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam')
callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='best_model1.h5', monitor='val_loss', save_best_only=True)]
print(model.summary())
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
history = model.fit(X_train, y_train,callbacks=callbacks, validation_data=(X_test, y_test),batch_size=128,epochs=13,verbose=2)

print("Test-acc:", np.mean(history.history["val_acc"]))

# plot train and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model train vs validation acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
