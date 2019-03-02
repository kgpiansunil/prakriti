# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:14:09 2019

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


train=pd.read_csv(os.getcwd()+'/prakriti_doc/trainingset.csv')
test=pd.read_csv(os.getcwd()+'/prakriti_doc/testset.csv')

from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
from scipy.stats import randint, uniform

cv = StratifiedKFold(y_train, n_folds=10, shuffle=True)

params_dist_grid = {
    'max_depth': [1, 5, 10],
    'gamma': [0, 0.5, 1],
    'n_estimators': randint(1, 1001), # uniform discrete random distribution
    'learning_rate': uniform(), # gaussian distribution
    'subsample': uniform(), # gaussian distribution
    'colsample_bytree': uniform(), # gaussian distribution
    'reg_lambda':uniform(),
    'reg_alpha':uniform()
    }

xgbc_fixed = {'booster':['gbtree'], 'silent':1}


bst_gridd = RandomizedSearchCV(estimator=XGBClassifier(*xgbc_fixed), param_distributions=params_dist_grid,\
                               scoring='accuracy', cv=cv, n_jobs=-1)

from sklearn.model_selection import train_test_split
seed = 123
x_data, x_test_data, y_data, y_test_data = train_test_split(X_train, y_train, test_size = 0.3,random_state=seed)

eval_set = [(x_test_data, y_test_data)]

XGBC = XGBClassifier(silent=1,n_estimators=641,learning_rate=0.2,max_depth=10,gamma=0.5,nthread=-1,\
                    reg_alpha = 0.05, reg_lambda= 0.35, max_delta_step = 1, subsample = 0.83, colsample_bytree = 0.6)


XGBC.fit(x_data, y_data, early_stopping_rounds=100, eval_set=eval_set, eval_metric='merror', verbose=True)

pred = XGBC.predict(x_test_data)

accuracy = accuracy_score(y_test_data, pred);
print ('accuracy:%0.2f%%'%(accuracy*100))

xgbc_pred= XGBC.predict(X_test)

 #saving to a csv file to make submission
solution = pd.DataFrame({'Id':df_Test.Id, 'Cover_Type':xgbc_pred}, columns = ['Id','Cover_Type'])
solution.to_csv('Xgboost_sol.csv', index=False)