import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

sys.path.append('/home/francis/Documents/ML/NIH ML')
from AutoClean.AutoClean import AutoClean

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
    
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

# https://towardsdatascience.com/automated-data-cleaning-with-python-94d44d854423

df = pd.read_csv("datasets/SPOTRIAS_organized_03May2022.csv")    
df = df.drop(pd.read_csv("datasets/features_to_drop.csv")['Columns'].tolist() , axis=1)  

col = df.pop("HOSP_MRSDIS")
df.insert(df.shape[1], col.name, col)
df = df.rename(columns = {'HOSP_MRSDIS':'label'}) 

from AutoClean import AutoClean
dfX = df.iloc[: , :-1] 
dfy = df.iloc[: , -1:]
pipeline = AutoClean(dfX)
df2 = pd.concat([pipeline.output, dfy], axis=1) 



df = pd.read_csv("datasets/SPOTRIAS_organized_03May2022.csv")
        
# select only rows where STRK_FINALDIAG = 1
df = df.loc[df['STRK_FINALDIAG'] == 2]

df = df.drop(pd.read_csv("datasets/features_to_drop.csv")['Columns'].tolist() , axis=1)  

label_name = 'HOSP_MRSDIS'
col = df.pop(label_name)
df.insert(df.shape[1], col.name, col)
df = df.rename(columns = {label_name :'label'}) 

from AutoClean import AutoClean
dfX = df.iloc[: , :-1] 
dfy = df.iloc[: , -1:]
pipeline = AutoClean(dfX)
df2 = pd.concat([pipeline.output, dfy], axis=1)     





        
dataTypeSeries = df.dtypes


df = df.head()
num_cols = df.select_dtypes('float').columns

missing_cols =  df[[num_cols]].columns[df[[num_cols]].isnull().any()].tolist() 

imputer = SimpleImputer(strategy='median', missing_values=np.nan)
imputer = imputer.fit(df[['B','C']])
df[['B','C']] = imputer.transform(df[['B','C']])


        
# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(df[:,:])
df[:,:] = imputer.transform(df[:,:])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_val = sc_X.transform(X_val)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(1,-1))
y_val = sc_y.fit_transform(y_val.reshape(1,-1))
y_test = sc_y.fit_transform(y_test.reshape(1,-1))

print ('*** Traning Data')
print (X_val)

print ('*** Testing Data')
print (X_test)

print ('*** Traning Output')
print (y_val)

print ('*** Testing Output')
print (y_test)