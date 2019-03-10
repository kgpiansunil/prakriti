# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 23:09:49 2019

@author: SUNIL
"""
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import os
import xgboost as xgb
import seaborn as sns
from xgboost.sklearn import XGBClassifier



train=pd.read_csv(os.getcwd()+'/prakriti_doc/trainingset.csv')
test=pd.read_csv(os.getcwd()+'/prakriti_doc/testset.csv')

list(train.columns.values)

index=test['index']
y=train['forest_cover_type']


described = train.describe()
train.isnull().sum()


sns.countplot(data=train,x=train['forest_cover_type'])
sns.boxplot(x="forest_cover_type", y="Elevation", data=train)
sns.boxplot(x="forest_cover_type", y="Aspect",data=train)


train=train.drop(['index','forest_cover_type'],1)
test=test.drop(['index'],1)

train[['Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways', 'Hillshade_Noon']] = np.sqrt(train[['Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_Noon']])
test[['Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways', 'Hillshade_Noon']] = np.sqrt(test[['Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_Noon']])

x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.3, random_state=42)


'''rf=RandomForestClassifier(n_estimators=300,class_weight='balanced',n_jobs=2,random_state=42)
rf.fit(x_train,y_train)
pred=rf.predict(x_test)
acc=rf.score(x_test,y_test)
print(acc)
rf.fit(train,y)
ct=rf.predict(test.iloc[:, :-1])
print(ct)'''




#--------------------ML Part Begins Here-------------#

#xgb_classifier = xgb.XGBClassifier(missing=np.nan, max_depth=7, n_estimators= 350, learning_rate =0.03, nthread=4, subsample = 0.95, colsample_bytree = 0.85, seed =4242)
#Parameter tuning
xgb_classifier = xgb.XGBClassifier(max_depth=15, n_estimators=100,max_features=0.25, min_samples_leaf=1, learning_rate=0.03)
xgb_classifier.fit(x_train, y_train)


#xgb_classifier.fit(X_train, y_train)
y_pred_X_test = xgb_classifier.predict(x_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_X_test)


#Cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=xgb_classifier, X = x_train, y = y_train, cv =10)
print(accuracies.mean())
print(accuracies.std())


#GridSearch
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
rf_para = [{'n_estimators':[50, 100], 'max_depth':[5,10,15], 'max_features':[0.1, 0.3], \
           'min_samples_leaf':[1,3], 'bootstrap':[True, False]}]


rfc = GridSearchCV(RandomForestClassifier(), param_grid=rf_para, cv = 10, n_jobs=-1)
rfc.fit(x_train, y_train)
rfc.best_params_
print ('Best accuracy obtained: {}'.format(rfc.best_score_))
print ('Parameters:')
for key, value in rfc.best_params_.items():
    print('\t{}:{}'.format(key,value))
    

eval_set = [(x_test, y_test)]
XGBC = XGBClassifier(silent=1,n_estimators=641,learning_rate=0.2,max_depth=10,gamma=0.5,nthread=-1,\
                    reg_alpha = 0.05, reg_lambda= 0.35, max_delta_step = 1, subsample = 0.83, colsample_bytree = 0.6)


XGBC.fit(x_train, y_train, eval_set=eval_set, eval_metric='merror', verbose=True)

pred = XGBC.predict(x_test_data)


#----------------------Output-------------------
output=pd.DataFrame(index)
output['forest_cover_type']=ct
output.head()
output.to_csv("output.csv",index=False)
