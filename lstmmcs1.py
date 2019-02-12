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
import string
import random
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#import tensorflow as tf
random.seed(7)

def preprocess_data(dataset, x_column, y_column):
    #x = dataset[x_column].map(lambda s: [ord(d) for d in s.encode('utf-8').decode('unicode_escape').encode('ascii','ignore').decode("utf-8")])
    y = dataset[y_column].values
    train_texts = [s.lower() for s in dataset[x_column]]

    # =======================Convert string to index================
    # Tokenizer
    tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
    tk.fit_on_texts(train_texts)
    # If we already have a character list, then replace the tk.word_index

    # construct a new vocabulary

    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1

    # Use char_dict to replace the tk.word_index
    tk.word_index = char_dict.copy()
    # Add 'UNK' to the vocabulary
    tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

    # Convert string to index
    train_sequences = tk.texts_to_sequences(train_texts)

    # Padding
    train_data = pad_sequences(train_sequences, maxlen=70, padding='post')

    # Convert to numpy array
    train_data = np.array(train_data, dtype='float32')
    return train_data, y

train = pd.read_csv('trainStrat.csv',engine='python')
X_train,y_train = preprocess_data(train,"text","HS")
test = pd.read_csv('testStrat.csv',engine='python')
X_test,y_test = preprocess_data(test,"text","HS")
#print(X[1498])

main_input = Input(shape=(70, ), name='main_input')
embedding = Embedding(input_dim=128, output_dim=128,input_length=70)(main_input)
lstm = LSTM(128, return_sequences=False)(embedding)
drop = Dropout(0.5)(lstm)
output = Dense(1, activation='sigmoid')(drop)
model = Model(inputs=main_input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam')
callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint(filepath='best_model1.h5', monitor='val_loss', save_best_only=True)]
print(model.summary())

history = model.fit(X_train, y_train,callbacks=callbacks, validation_data=(X_test, y_test),batch_size=128,epochs=30,verbose=2)

print("Test-acc:", np.mean(history.history["val_acc"]))

# plot train and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show();

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model train vs validation acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show();
