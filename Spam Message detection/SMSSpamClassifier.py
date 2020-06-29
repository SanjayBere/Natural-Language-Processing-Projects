# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:53:35 2020

@author: GCS
"""

import pandas as pd

#Load the Data 
Messages = pd.read_csv("D:/AppliedAICourse/Projects/Dataset/smsspamcollection/SMSSpamCollection",sep = "\t",names=['labels','Message'])

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
ps = PorterStemmer()
corpus = []

# Data Preprocessing
for i in range(0,len(Messages)):
    sentence = re.sub('[^a-zA-Z]',' ',Messages['Message'][i])
    sentence = sentence.lower()
    sentence = sentence.split()
    
    words = [ps.stem(word) for word in sentence if word not in stopwords.words("english")]
    words = ' '.join(words)
    corpus.append(words)
    

#Creating the BOW

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(Messages['labels'])
y = y.iloc[:,0].values

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.20 , random_state = 0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train,)

y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test , y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)
