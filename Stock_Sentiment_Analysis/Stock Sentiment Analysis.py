# -*- coding: utf-8 -*-

#laod the data 
import pandas as pd 
import re

df = pd.read_csv("D://AppliedAICourse//Projects//NLP//Stock Sentiment Analysis//Stock News Headlines.csv",encoding="ISO-8859-1")
df.head(100)

#We are splitting are data using Time Based Splitting 
train = df[df["Date"] <  '20150101']    #data is in the format of yyyy-dd-mm
print("------train Dataset lenght-----\n",train.shape)
test = df[df["Date"] > '20141231']
print("------test Dataset-----\n",test.shape)

data = train.iloc[:,2:27]
#print(data)
data.replace("[^a-zA-Z]"," ",regex=True,inplace = True)

#Renaming Columns for ease
list1 = [i for i in range(25)]
new_Index = [str(i) for i in list1]  #Since i string , as it is not compatible , so we are converting into string format
data.columns = new_Index
data.head(5)

for index in new_Index:
    #print(data[index])
    data[index] = data[index].str.lower()  # data[0] -> 0th row -> since it is a string type object 
data.head()

headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

print(headlines[3])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
countVect = CountVectorizer(ngram_range=(3,3))
Traindataset = countVect.fit_transform(headlines)
randomClassifier = RandomForestClassifier(n_estimators=200,criterion='entropy')
randomClassifier.fit(Traindataset,train['Label'])

TestData = []
for row in range(0,len(test.index)):
    TestData.append(' '.join(str(x) for x in test.iloc[row,2:27]))    
test_dataset = countVect.transform(TestData)
pred = randomClassifier.predict(test_dataset)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
matrix = confusion_matrix(test['Label'],pred)
print(matrix)
acc = accuracy_score(test['Label'],pred)
print("="*127)
print(acc)
report = classification_report(test['Label'],pred)
print("="*127)
print(report)
