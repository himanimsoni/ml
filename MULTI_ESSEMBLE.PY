
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
#random fix
seed = 7
np.random.seed(seed)

tweets_data = pd.read_csv("pd_misogyny_testing_csv.csv")

X = tweets_data['text']
y = tweets_data['misogyny_category']
accuracy_scores = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)


# create the sub models
estimators = []
model1 = LogisticRegression(penalty='l1',C=0.7, n_jobs=1,multi_class='multinomial',solver='saga')
estimators.append(('logistic', model1))
model2 = clf = RandomForestClassifier(n_estimators = 250, random_state = 0,criterion='entropy')
estimators.append(('rf', model2))
model3 = MultinomialNB()
estimators.append(('nb', model3))
model4 = SGDClassifier(max_iter =5, alpha=0.01, loss='modified_huber')
estimators.append(('sgd', model4))
model5 = XGBClassifier(n_estimators = 200, max_depth = 25)
estimators.append(('xboost', model5))


# create the ensemble model
ensemble = VotingClassifier(estimators,voting='soft')
ensemble = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3),stop_words='english',max_df=1.5,min_df=4)),('ensemble', ensemble)])
results = model_selection.cross_val_score(ensemble, X_train, y_train, cv=10, scoring='f1_macro')
print(results)
print(results.mean())

ensemble = VotingClassifier(estimators,voting='soft')
ensemble = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3),stop_words='english',max_df=1.5,min_df=4)),('ensemble', ensemble)])
ensemble = ensemble.fit(X_train, y_train)
y_predicted = ensemble.predict(X_test)
print(y_predicted)

# Reporting on classification performance)
print("f1_score categorys: %.f" %np.average(f1_score(y_test,y_predicted, average=None)))
print(classification_report(y_test,y_predicted))
accuracy_scores.append(np.average(f1_score(y_test,y_predicted, average=None)))
classes = ['discredit','sexual_harassment','stereotype','dominance','derailing']
cnf_matrix = confusion_matrix(y_test,y_predicted,labels=classes)
print("Confusion matrix:")
print(cnf_matrix)

'''

def predictFromData(X_train, X_test, y_train, y_test):
    ensemble = VotingClassifier(estimators,voting='soft')
    ensemble = Pipeline([('vect', CountVectorizer()),('ensemble', ensemble)])
    ensemble = ensemble.fit(X_train, y_train)
    y_predicted = ensemble.predict(X_test)
    print(y_predicted)

    print("Accuracy: %.2f" %accuracy_score(y_test,y_predicted))
    classes = [0,1]
    cnf_matrix = confusion_matrix(y_test,y_predicted,labels=classes)
    print("Confusion matrix:")
    print(cnf_matrix)

    pass

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.10)
for train, test in sss.split(X,y):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    #print(len(X_train), len(X_test), len(y_train), len(y_test))
    predictFromData(X_train,X_test,y_train,y_test)
    #print("X_train")
    #print(X_train)
    #print("X_test")
    #print(X_test)
    #print("y_train")
    #print(y_train)
    #print("y_test")

print('Accuracy', np.mean(accuracy_scores))
'''
