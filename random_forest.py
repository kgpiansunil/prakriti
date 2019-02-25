# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 23:09:49 2019

@author: SUNIL
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from subprocess import check_output
import os
#import seaborn as sns

train=pd.read_csv(os.getcwd()+'/prakriti_doc/trainingset.csv')
test=pd.read_csv(os.getcwd()+'/prakriti_doc/testset.csv')

list(train.columns.values)

index=test['index']
y=train['forest_cover_type']

train.describe()
train.isnull().sum()

sns.countplot(data=train,x=train['Cover_Type'])
sns.boxplot(x="Cover_Type", y="Elevation", data=train)
sns.boxplot(x="Cover_Type", y="Aspect",data=train)

train=train.drop(['index','forest_cover_type'],1)
test=test.drop(['index'],1)

x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.3, random_state=42)

rf=RandomForestClassifier(n_estimators=300,class_weight='balanced',n_jobs=2,random_state=42)

rf.fit(x_train,y_train)

pred=rf.predict(x_test)

acc=rf.score(x_test,y_test)
print(acc)

rf.fit(train,y)
ct=rf.predict(test.iloc[:, :-1])
print(ct)

output=pd.DataFrame(index)
output['forest_cover_type']=ct
output.head()

output.to_csv("output.csv",index=False)

