import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from itertools import groupby
import pickle
from sys import argv
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
from keras.optimizers import SGD
from keras.layers import Dropout
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

tweets_data = pd.read_csv("en_training_clean.csv")
en_tweets_data = tweets_data.loc[:,['text','misogynous']]


# Preparing the train and test data
X = en_tweets_data['text']
y = en_tweets_data['misogynous']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# Training a Naive Bayes model
count_vect = CountVectorizer()
count_vect_result_train= count_vect.fit_transform(X_train)
count_vect_X_train = pd.DataFrame(count_vect_result_train.todense(), columns=count_vect.get_feature_names())
count_vect_X_train_final = count_vect_X_train.iloc[:,:]
#print("X_train")
#print(X_train)
#print("X_test")
#print(X_test)
#print("y_train")
#print(y_train)
#print("y_test")
#print(y_test)
model = Sequential()
model.add(Dropout(0.8,input_shape=(9289,)))
model.add(Dense(2500, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(620, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(155, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(1, activation='sigmoid'))
'''

model = Sequential()
model.add(Dropout(0.8,input_shape=(9289,)))
model.add(Dense(units = 2500, activation = "relu"))
model.add(Dropout(0.8))
model.add(Dense(units = 620, activation = "relu"))
model.add(Dropout(0.8))
model.add(Dense(units = 155, activation = "relu"))
model.add(Dropout(0.8))
model.add(Dense(units = 1, activation = "sigmoid"))
model.save('mlp.h5')
'''
'''
model = Sequential()
model.add(Dropout(0.5,input_shape=(9921,)))
model.add(Dense(2500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(620, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(155, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


model = Sequential()
model.add(Dropout(0.5,input_shape=(9921,)))
#model.add(Dropout(0.8,input_shape=(9921,)))
model.add(Dense(2500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(124, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
'''

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(count_vect_X_train_final,y_train,epochs=20, batch_size=20)

X_test = count_vect.transform(X_test)
count_vect_X_test = pd.DataFrame(X_test.todense(), columns=count_vect.get_feature_names())
count_vect_X_test_final = count_vect_X_test.iloc[:,:]
y_predicted = model.predict(X_test)
#print(count_vect_X_test_final)
#history = model.fit(count_vect_X_train_final,y,epochs=20, batch_size=20, validation_split=0.15)

accuracy_score = accuracy_score(y_test,y_predicted.round())
print("Accuracy: %.2f" %accuracy_score)
classes = [0,1]
cnf_matrix = confusion_matrix(y_test,y_predicted.round(),labels=classes)
print("Confusion matrix:")
print(cnf_matrix)
'''
print("Test-Accuracy:", np.mean(history.history["val_acc"]))
# plot train and validation loss
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.title('model train vs validation accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()
'''
