# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 21:21:23 2021

@author: bharadwaj
"""

import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("D:/tinku/outstanding/healthcare-dataset-stroke-data.csv")

data.head(5)

data.dtypes

cat_columns=['gender','ever_married','work_type','Residence_type','smoking_status']
le=LabelEncoder()
for i in cat_columns:
    data[i]=le.fit_transform(data[i])

data.dtypes

data.isnull().sum()

data['bmi'].fillna(0,inplace=True)
data.isnull().sum()

y=data['stroke']
x=data.drop('stroke',axis=1)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
clf_xgb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
clf_rf= RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2)
clf_ada =AdaBoostClassifier(n_estimators=100)
clf_xgb.fit(xtrain,ytrain)
xgb_pred=clf_xgb.predict(xtest)
clf_rf.fit(xtrain,ytrain)
rf_pred=clf_rf.predict(xtest)
clf_ada.fit(xtrain,ytrain)
ada_pred=clf_ada.predict(xtest)

print(f'The Accuracy Score for xgb is {round(accuracy_score(ytest,xgb_pred)*100,4)}%')
print(f'The Accuracy Score for Random Forest is {round(accuracy_score(ytest,rf_pred)*100,4)}%')
print(f'The Accuracy Score for Ada Boost is {round(accuracy_score(ytest,ada_pred)*100,4)}%')

########## we can say that these are the different accuracies for different techniques