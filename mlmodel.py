import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, render_template,request
import pickle #Initialize the flask App
from sklearn import preprocessing
data=pd.read_csv('dataset.csv')
data=data.drop('Unnamed: 0',axis=1)
print ("Dataset Length: ", len(data)) 
print ("Dataset Shape: ", data.shape)
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder() 
data.final_result=encoder.fit_transform(data.final_result)
X=data.drop('Engagement',axis=1) #make changes
y=data['Engagement']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
pickle.dump(gnb, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))