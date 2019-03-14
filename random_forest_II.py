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
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from scipy.stats import randint, uniform


train=pd.read_csv(os.getcwd()+'/prakriti_doc/trainingset.csv')
test=pd.read_csv(os.getcwd()+'/prakriti_doc/testset.csv')

df_train1 = train
df_test1 = test
df_train1['Horizontal_Distance_To_Hydrology'] = np.sqrt(df_train1['Horizontal_Distance_To_Hydrology'])
df_train1['Horizontal_Distance_To_Roadways'] = np.sqrt(df_train1['Horizontal_Distance_To_Roadways'])
df_train1['Hillshade_Noon'] = np.sqrt(df_train1['Hillshade_Noon'])
df_test1[['Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways', 'Hillshade_Noon']] = np.sqrt(df_test1[['Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_Noon']])


#Create two new columns named Slope hydrology and Slope hydrology percent and remove any infinite values that may result 
df_train1['slope_hyd'] = np.sqrt(df_train1.Vertical_Distance_To_Hydrology**2 + \
        df_train1.Horizontal_Distance_To_Hydrology**2) 
df_train1.slope_hyd=df_train1.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) 

df_test1['slope_hyd'] = np.sqrt(df_test1.Vertical_Distance_To_Hydrology**2 + \
        df_test1.Horizontal_Distance_To_Hydrology**2) 

'''#Elevation adjusted by Horizontal distance to Hyrdrology
df_train1['Elev_to_HD_Hyd']=df_train1.Elevation - 0.2 * df_train1.Horizontal_Distance_To_Hydrology
df_train1['Elev_to_HD_Road']=df_train1.Elevation - 0.05 * df_train1.Horizontal_Distance_To_Roadways
df_train1['Elev_to_VD_Hyd']=df_train1.Elevation - 0.05 * df_train1.Vertical_Distance_To_Hydrology

df_train1['Mean_Amenities']=(df_train1.Horizontal_Distance_To_Hydrology + df_train1.Horizontal_Distance_To_Roadways) / 2
'''

cols=df_train1.columns.tolist()
cols=cols[1:11]+cols[12:]+cols[11:12] 
df_train1=df_train1[cols] 

cols=df_test1.columns.tolist()
cols=cols[1:11]+cols[12:]+cols[11:12] 
df_test1=df_test1[cols] 

X_train = df_train1.iloc[:, :-1]
y_train = df_train1.iloc[:, -1:].values

X_test = df_test1.iloc[:, :-1]

'''cv = StratifiedKFold(shuffle=True, n_splits=10)

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


####

xgb_classifier = xgb.XGBClassifier(max_depth=15, n_estimators=100,max_features=0.25, min_samples_leaf=1, learning_rate=0.03)
xgb_classifier.fit(x_data, y_data)


#xgb_classifier.fit(X_train, y_train)
y_pred_X_test = xgb_classifier.predict(x_test_data)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_data, y_pred_X_test)


#Cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=rf, X = y_test_data, y = pred, cv =10)
print(accuracies.mean())

#GridSearch
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
rf_para = [{'n_estimators':[50, 100], 'max_depth':[5,10,15], 'max_features':[0.1, 0.3], \
           'min_samples_leaf':[1,3], 'bootstrap':[True, False]}]


rfc = GridSearchCV(RandomForestClassifier(), param_grid=rf_para, cv = 10, n_jobs=-1)
rfc.fit(x_data, y_data)
rfc.best_params_
print ('Best accuracy obtained: {}'.format(rfc.best_score_))
print ('Parameters:')
for key, value in rfc.best_params_.items():
    print('\t{}:{}'.format(key,value))
 '''   
  #best results
from sklearn.metrics import roc_auc_score  
rf=RandomForestClassifier(bootstrap = True,min_samples_leaf=1,max_features = 0.3, max_depth = 5, n_estimators=200,class_weight='balanced',n_jobs=2,random_state=42)
clf = rf.fit(X_train,y_train)
pred= rf.predict(X_test)
results = clf.predict_proba(X_test)
#acc = roc_auc_score(y_test_data, pred, average='macro', sample_weight=None, max_fpr=None)
 #best reuslts 
solution = pd.DataFrame({'index':test.iloc[:,0:1], 'forest_cover_type':results[:,1]}, columns = ['index','forest_cover_type'])
solution.to_csv('Xgboost_sol.csv', index=False)

ab = [test.iloc[:,0:1],results[:,1] ]
df = pd.DataFrame(data =results[:,1]  ,columns=['Prob_belonging_to_1st_category'])

t = test.iloc[:, :-1]
data = t.join(df)

data = data.iloc[:, [0, 11]]

data.to_csv('submission.csv', index=False)


 '''
acc=rf.score(y_test_data,pred)
print(acc)
rf.fit(X_train,y_train)
ct=rf.predict(test.iloc[:, :-1])
print(ct)

from sklearn.metrics import roc_auc_score
#acc = roc_auc_score(y_test_data, results[:,1], average='macro', sample_weight=None, max_fpr=None)
acc = roc_auc_score(y_test_data, pred, average='macro', sample_weight=None, max_fpr=None)


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(x_data, y_data)

rf_random.best_params_


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)
base_model.fit(x_data, y_data)
base_accuracy = evaluate(base_model, x_test_data, y_test_data)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, x_test_data, y_test_data)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [False],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [1,2,3],
    'min_samples_split': [3, 5, 7],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(x_data, y_data)

grid_search.best_params_

best_grid = grid_search.best_estimator_
print(best_grid)

rf=RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=110, max_features=3, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=3, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
rf=RandomForestClassifier(bootstrap = False,min_samples_leaf=3, min_samples_split=5,max_features = 3,criterion='gini', max_depth =110, n_estimators=100,class_weight=None,n_jobs=2,random_state=42)

clf = rf.fit(x_data,y_data)
pred= rf.predict(x_test_data)
results = clf.predict_proba(x_test_data)
acc = roc_auc_score(y_test_data, pred, average='macro', sample_weight=None, max_fpr=None)
'''