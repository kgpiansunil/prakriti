# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:14:39 2019

@author: SUNIL
"""

# Common libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Restrict minor warnings
import warnings
warnings.filterwarnings('ignore')

df_train=pd.read_csv(os.getcwd()+'/prakriti_doc/trainingset.csv')
df_Test =pd.read_csv(os.getcwd()+'/prakriti_doc/testset.csv')
df_test=df_Test 

df_train.head()

described= df_train.describe()
df_train.isnull().sum()

#df_train=df_train.drop(['index','forest_cover_type'],1)
df_train=df_train.drop(['index'],1)
df_test=df_test.drop(['index'],1)


#correlation matrix
sns.heatmap(df_train.corr(),vmax=0.8,square=True)
data = df_train.iloc[:,:]
# Get name of the columns
cols = data.columns
# Calculate the pearson correlation coefficients for all combinations
data_corr = data.corr()
# Threshold ( only highly correlated ones matter)
threshold = 0.3
corr_list = []
size =11
# Sorting out the highly correlated values
for i in range(0, size):
    for j in range(i+1, size):
        if data_corr.iloc[i,j]>= threshold and data_corr.iloc[i,j]<1\
        or data_corr.iloc[i,j] <0 and data_corr.iloc[i,j]<=-threshold:
            corr_list.append([data_corr.iloc[i,j],i,j])
        
# Sorting the values
s_corr_list = sorted(corr_list,key= lambda x: -abs(x[0]))

# print the higher values
for v,i,j in s_corr_list:
    print("%s and %s = %.2f" % (cols[i], cols[j], v))
    
df_train.skew()

sns.pairplot(data = df_train, hue='forest_cover_type', size= 6, x_vars=cols[5], y_vars=cols[3])

#df_train.Neota.value_counts() class distinction easily be seen
#1 spruce/fir 0 load polepine


#Horizontal_Distance_To_Hydrology
df_train1 = df_train
from scipy import stats
plt.figure(figsize=(8,6))
sns.distplot(df_train1['Horizontal_Distance_To_Hydrology'], fit = stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Horizontal_Distance_To_Hydrology'], plot=plt)

df_train1['Horizontal_Distance_To_Hydrology'] = np.sqrt(df_train1['Horizontal_Distance_To_Hydrology'])

# Plot again after sqrt transformation
plt.figure(figsize=(8,6))
sns.distplot(df_train1['Horizontal_Distance_To_Hydrology'], fit = stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Horizontal_Distance_To_Hydrology'], plot=plt)


#Vertical_Distance_To_Hydrology
plt.figure(figsize=(8,6))
sns.distplot(df_train1['Vertical_Distance_To_Hydrology'], fit = stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Vertical_Distance_To_Hydrology'], plot=plt)

#Horizontal_Distance_To_Roadways
plt.figure(figsize=(8,6))
sns.distplot(df_train1['Horizontal_Distance_To_Roadways'], fit=stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Horizontal_Distance_To_Roadways'], plot=plt)

df_train1['Horizontal_Distance_To_Roadways'] = np.sqrt(df_train1['Horizontal_Distance_To_Roadways'])

# Plot again after sqrt transformation
plt.figure(figsize=(8,6))
sns.distplot(df_train1['Horizontal_Distance_To_Roadways'], fit = stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Horizontal_Distance_To_Roadways'], plot=plt)

#Hillshade_9am
fig = plt.figure(figsize=(8,6))
sns.distplot(df_train1['Hillshade_9am'],fit=stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Hillshade_9am'],plot=plt)


# Hillshade_Noon
fig = plt.figure(figsize=(8,6))
sns.distplot(df_train1['Hillshade_Noon'],fit=stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Hillshade_Noon'],plot=plt)

df_train1['Hillshade_Noon'] = np.sqrt(df_train1['Hillshade_Noon'])

# Plot again after square transformation
fig = plt.figure(figsize=(8,6))
sns.distplot(df_train1['Hillshade_Noon'],fit=stats.norm)
fig = plt.figure(figsize=(8,6))
res = stats.probplot(df_train1['Hillshade_Noon'],plot=plt)

# To be used in case of algorithms like SVM
df_test1 = df_test
df_test1[['Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways', 'Hillshade_Noon']] = np.sqrt(df_test1[['Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways',,'Hillshade_Noon']])

from sklearn.preprocessing import StandardScaler


