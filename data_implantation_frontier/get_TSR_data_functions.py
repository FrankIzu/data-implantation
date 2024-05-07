#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 17:03:41 2021

@author: Francis
"""
import os
import sys
import csv

import warnings
warnings.filterwarnings("ignore")

#sys.path.append('AutoClean') # append AutoClean folder to current path
from .AutoClean.AutoClean import AutoClean

import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
    
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans
import sklearn.cluster as cluster
from sklearn.metrics import silhouette_score
from statistics import mode

from sklearn.feature_selection import SelectKBest, f_classif, chi2

import pandas as pd
import numpy as np
import hdbscan

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

from .get_TSR_DI_functions import TSR_DI_functions

from loguru import logger
      

class TSR_data_functions:

    def get_TSR_imp(name):
        name = "Front_implanted"
        
        if name=="raw":  
            df_tr = pd.read_csv("datasets/wholeset_Jim_nomissing_validated.csv")
            df_tr = df_tr.reindex(sorted(df_tr.columns), axis=1).drop(['IDCASE_ID','ICASE_ID','MRS_1','MRS_3'], axis=1)   
            col = df_tr.pop("discharged_mrs")
            df_tr.insert(df_tr.shape[1], col.name, col)
            df_tr.rename(columns = {'discharged_mrs':'label'}) 
            
            X = df_tr.iloc[:,0:-1].values
            Y = df_tr.iloc[:,-1].values   
            
            norm = MinMaxScaler().fit(X)
            X = norm.transform(X)
                
            encoder = LabelEncoder()
            encoder.fit(Y)
            encoded_Y = encoder.transform(Y)
            # convert integers to dummy variables (i.e. one hot encoded)
            Y = to_categorical(encoded_Y)
        
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state = 1)
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = 1)
            
        else:
            
             # section for training 
            df_train = pd.read_csv("datasets/TSR_"+ name + ".csv").sample(frac=1).reset_index(drop=True)
            
            X_train = df_train.iloc[:,0:-1].values
            y_train = df_train.iloc[:,-1].values   
            
            norm = MinMaxScaler().fit(X_train)
            X_train = norm.transform(X_train)
                
            encoder = LabelEncoder()
            encoder.fit(y_train)
            encoded_Y_train = encoder.transform(y_train)
            # convert integers to dummy variables (i.e. one hot encoded)
            dummy_y_train = to_categorical(encoded_Y_train)
            y_train = dummy_y_train
            
            
            # section for testing
            df = pd.read_csv("datasets/wholeset_Jim_nomissing_validated.csv")  
           
            df = df.reindex(sorted(df.columns), axis=1).drop(['IDCASE_ID','ICASE_ID','MRS_1','MRS_3'], axis=1)   
            col = df.pop("discharged_mrs")
            df.insert(df.shape[1], col.name, col)
            df = df.rename(columns = {'discharged_mrs':'label'})   
            
            X = df.iloc[:,0:-1]
            Y = df.iloc[:,-1]  
            _, i_X_test, _, i_y_test = train_test_split(X, Y, test_size = 0.4, random_state = 1)
            
            df_test_val = np.column_stack((i_X_test, i_y_test))        
              
                  
            X_test_val = df_test_val[:,0:-1]
            y_test_val = df_test_val[:,-1]
            
            norm = MinMaxScaler().fit(X_test_val)
            X = norm.transform(X_test_val)
                
            encoder = LabelEncoder()
            encoder.fit(y_test_val)
            encoded_y_test_val = encoder.transform(y_test_val)
            # convert integers to dummy variables (i.e. one hot encoded)
            y_test_val = to_categorical(encoded_y_test_val)
    # =============================================================================
    #         
    #         # run occassionally
    #         norm = MinMaxScaler().fit(X_train)
    #         X_train = norm.transform(X_train)
    #         encoded_y_train = encoder.transform(y_train)
    #         y_train = to_categorical(encoded_y_train)
    # =============================================================================
                
            X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size = 0.5, random_state = 1)
              
            #logger.info(np.unique(Y))
        
        return X_train, X_test, X_val, y_train, y_test, y_val, len(np.unique(Y))
        
    def get_TSR():
        df = pd.read_csv("datasets/wholeset_Jim_nomissing_validated.csv")    
        
        df = df.reindex(sorted(df.columns), axis=1).drop(['IDCASE_ID','ICASE_ID','MRS_1','MRS_3'], axis=1)    
          
        col = df.pop("discharged_mrs")
        df.insert(df.shape[1], col.name, col)
    
        df = df.rename(columns = {'discharged_mrs':'label'}) 
        
        X = df.iloc[:,0:-1].values
        Y = df.iloc[:,-1].values                  
         
        norm = MinMaxScaler().fit(X)
        X = norm.transform(X)
            
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = to_categorical(encoded_Y)
    
        X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.4, random_state = 1)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = 1)
        
        return X_train, X_test, X_val, y_train, y_test, y_val, len(np.unique(Y))
    
    def get_clean_TSR():
        df = pd.read_csv("find_outliers/TSR_IF.csv")    
            
        X = df.iloc[:,0:-1].values
        Y = df.iloc[:,-1].values                  
         
        norm = MinMaxScaler().fit(X)
        X = norm.transform(X)
            
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = to_categorical(encoded_Y)
    
        X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.4, random_state = 1)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = 1)
        
        return X_train, X_test, X_val, y_train, y_test, y_val, len(np.unique(Y))
    
    def get_TSR_by_Class_All(self):        
        # This function should not be used becasue it uses ALL samples for DI, which is NOT RIGHT!
        # Some samples are supposed to be reserved for testing and validation
        df = pd.read_csv("datasets/wholeset_Jim_nomissing_validated.csv")    
        
        df = df.reindex(sorted(df.columns), axis=1).drop(['IDCASE_ID','ICASE_ID','MRS_1','MRS_3'], axis=1)    
          
        col = df.pop("discharged_mrs")
        df.insert(df.shape[1], col.name, col)
    
        df = df.rename(columns = {'discharged_mrs':'label'}) 
        
        X = df.iloc[:,0:-1].values
        Y = df.iloc[:,-1].values   
        
        df_tr = np.column_stack((X, Y))   
        
        # convert back to dataframe
        df_tr = pd.DataFrame(df_tr, columns = [f's{x+1}' for x in range(df_tr.shape[1])])  
        df_tr = df_tr.rename(columns = {df_tr.columns[len(df_tr.columns)-1]:'label'}) 
       
        #logger.info(df_tr.label.unique())
        data0 = df_tr[df_tr['label'] == 0].reset_index(drop=True)
        data1 = df_tr[df_tr['label'] == 1].reset_index(drop=True)
        data2 = df_tr[df_tr['label'] == 2].reset_index(drop=True)
        data3 = df_tr[df_tr['label'] == 3].reset_index(drop=True)
        data4 = df_tr[df_tr['label'] == 4].reset_index(drop=True)
        data5 = df_tr[df_tr['label'] == 5].reset_index(drop=True)
        
        #logger.info(df.label.unique())
                       
        return [data0, data1, data2, data3, data4, data5], len(np.unique(df_tr.iloc[:,-1].values))   

    def doPCA_selection(self, X, y, size, target=''):
        di = TSR_DI_functions()
        
        n_comp = size if min(X.shape[0], X.shape[1]) > size else min(X.shape[0], X.shape[1])
        X_pca, _, dictn, explained_variance = di.doPCA(X, size)  
        # 1st way to get the list
        vector_names = list(dictn.values())[0:n_comp]
        
        if target:
            X_pca = np.column_stack((X_pca, X[target]))
            vector_names.append(target)
        vector_names.append('label')
        
        X_pca = pd.DataFrame(data = np.column_stack((X_pca, y)), columns = vector_names)   
        
        return X_pca, vector_names
    
    def doFeature_selection(self, X, y, size, target=''):
        selector = SelectKBest(score_func=f_classif, k=size)
        selector.fit(X, y)
        
        X_new = selector.transform(X)        
        X.columns[selector.get_support(indices=True)]
                   
        
        # 1st way to get the list
        vector_names = list(X.columns[selector.get_support(indices=True)])
        
        if target:
            X_new = np.column_stack((X_new, X[target]))
            vector_names.append(target)
        vector_names.append('label')
        #2nd way
        X.columns[selector.get_support(indices=True)].tolist()
        X_new = pd.DataFrame(data = np.column_stack((X_new, y)), columns = vector_names)   
        
        return X_new, vector_names
    
    def preprocess(self, df):
        pipeline = AutoClean(df, outliers=False)        
        d = pipeline.output  
        return d
    
    @staticmethod
    def np_to_df(numarr):
        # convert back to dataframe
        df = pd.DataFrame(numarr, columns = [f's{x}' for x in range(numarr.shape[1])])  
        df = df.rename(columns = {df.columns[len(df.columns)-1]:'label'}) 
        return df
    
    def get_no_of_clusters_dbscan(self, data): # not used
        clusterer = hdbscan.HDBSCAN(max_cluster_size=5)
        clusterer.fit(data)
        logger.info('Number of clusters found = {}'.format(clusterer.labels_.max() + 1))
        return clusterer.labels_.max() + 1 # hdbscan count starts from 0
    
    def get_no_of_clusters_kmeans(self, data):
        limit = int((data.shape[0]//2)**0.5) 
        # determining number of clusters using silhouette score 
        max_k = 0  
        max_score = 0
                
        for k in range(2, limit+1):
            model = KMeans(n_clusters=k)
            model.fit(data)
            pred = model.predict(data)
            score = silhouette_score(data, pred)
            if score > max_score:
                max_score = score
                max_k = k
            #logger.info('Silhouette Score for k = {}: {:<.3f}'.format(k, score))
        return max_k
      
    
    def clean_dataset(self, df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)
        
    def get_TSR_by_Class_generic(self, dataset, class_size, colname, nfeatures, labelname, testset): 
        df = dataset
       
        target_label = labelname # HOSP_MRSDIS HOSP_MRS90
        # df = df.loc[(df['STRK_FINALDIAG'] == 1) & (df[target_label] != 91) & (df[target_label] != 99) & df[target_label].notnull()]
        
        # df['STRK_FINALDIAG'] = df['STRK_FINALDIAG'].astype(int)
        # df[target_label] = df[target_label].astype(int)
        
        # # makind exception for HIST_CEREB_ISCH
        # df = df.loc[(df['HIST_CEREB_ISCH'] == str('Y')) | (df['HIST_CEREB_ISCH'] == str('N'))]               
        # df.loc[df['HIST_CEREB_ISCH'] == 'Y', 'HIST_CEREB_ISCH'] = 1 # recurrent stroke
        # df.loc[df['HIST_CEREB_ISCH'] == 'N', 'HIST_CEREB_ISCH'] = 0 # first time stroke
        # df['HIST_CEREB_ISCH'] = pd.to_numeric(df['HIST_CEREB_ISCH'])
        
        # df = df.loc[(df['HIST_CEREB_ISCH'] == 1)]
       
        col = df.pop(target_label)
        df.insert(df.shape[1], col.name, col)
        
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        
        #logger.info(df.columns)
        dfX = df.iloc[: , :-1] 
        dfy = df.iloc[: , -1:]
        p = self.preprocess(dfX)
        #logger.info(f'after {d.columns}')
        df = pd.concat([p.reset_index(drop=True), dfy.reset_index(drop=True)], axis=1)  
       
        df = df.select_dtypes(exclude=["object"]).astype(float)
                       
        #To get ride of all Unnamed columns, you can also use regex
        df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
        #logger.info(df.dtypes)
        df.to_csv('datasets/mydataset_clean.csv', index=False)  
                    
        #logger.info(np.isfinite(df.all())) #and gets True
                
        df = df.rename(columns = {target_label:'label'}) 
        #df = df.drop(pd.read_csv("datasets/features_to_drop.csv")['Columns'].tolist() , axis=1, errors='ignore')                  
                         
        #logger.info(df.isnull().sum())
        
        dfcolumns = df.columns
         
        di = TSR_DI_functions()
        working_dir = os.getcwd() + "/datasets/"
        #pd.DataFrame(features).to_csv(working_dir + '/iFeatures' + ".csv", index=None)
        target = 0
        no_pca = 20
        unique_names = []
        with_fs = True
        
        if nfeatures == -1:
            with_fs = False
            print()
            logger.warning('NOTE: Program execution without feature selection is activated')
            print()
        elif nfeatures < 3:
            sys.exit('Program execution halted. Number of features must be greater than two')
            logger.critical('Program execution halted. Number of features must be greater than two')
                
        if colname == 'NoTarget':              
            
            df_tr = df

            if testset:
                X = df.iloc[:,0:-1].values
                Y = df.iloc[:,-1].values   
                i_X_train, i_X_test, i_y_train, i_y_test = train_test_split(X, Y, test_size = 0.4, random_state = 1)   
                
                # save the test set. If you rebalance entire dataset, you can't use the dataset for testing otherwise you will have bias in your result
                data_test = pd.DataFrame(data = np.column_stack((i_X_test, i_y_test)), columns = dfcolumns)
                logger.info(f'Testset (40%) data was saved and will not be used in Data Implanation ')
                data_test.to_csv('datasets/test dataset.csv', index=False) 
                
                df_tr = pd.DataFrame(data = np.column_stack((i_X_train, i_y_train)), columns = dfcolumns)
            
            
            #logger.info(df_tr.label.unique())
            
            data = []
            diction = {}  
            parent_dir = working_dir + colname
            
            if with_fs == True:
                # pass X and y as parameters for feature selection
                df2, cols = self.doFeature_selection(df_tr.drop('label', axis=1), df_tr.pop('label'), nfeatures) 
            else:
                df2, cols = df_tr, dfcolumns.tolist() 
            
            logger.info(f'Now creating {colname} workspace .....')
            for index in range(class_size):  
                df3 = df2[df2['label'] == index].reset_index(drop=True)
                label = df3.iloc[:,-1] 
                s_df = df3.drop('label', axis=1)               
                
                # directory = str(index)
                # # save each sub-sub population into a seperate file
                # path = os.path.join(parent_dir, directory)   
                # try:
                #     os.makedirs(path, exist_ok = True)
                #     logger.info("Directory '%s' created successfully" % directory)
                #     # if no of features is provided, make sure that no is greater than the number of PCA we want to do otherwise, use the number of PCA as teh no of features
                #     if (nfeatures < no_pca):
                #         nfeatures = no_pca                      
                        
                #     # doPCA and discard other data not needed and then save the PCA columns
                #     # n_components must be between 0 and min(n_samples, n_features)
                #     n_comp = nfeatures if min(s_df.shape[0], s_df.shape[1]) > nfeatures else min(s_df.shape[0], s_df.shape[1])
                #     #logger.info(s_df.shape)
                #     _, _, dictn, explained_variance = di.doPCA(s_df, n_comp)    
                                                
                #     evariance = list(explained_variance)
                #     ev_name = list(dictn.values())
                #     # merge the PCA info together for storate. Get first no_pca elements 
                #     pd.DataFrame(np.column_stack((np.array(ev_name[0:no_pca]).flatten(), np.round(explained_variance[0:no_pca], 5)))).to_csv(path + '/evariance' + ".csv", header=['feature', 'ratio'], index=None)                     
                    
                #     #save sub-sub population
                #     pd.DataFrame(df3).to_csv(path + '/' + str(index) + ".csv", index=None)
                # except OSError as error:
                #     logger.info("Directory '%s' can not be created" % directory)
                
                s_data = np.column_stack((s_df, label))
                                                              
                data.append(self.np_to_df(s_data))
            
            # HOSP_MRSDIS must always be part of the dataset --- its the label!!!
            cols[cols.index('label')] = target_label
                        
            # save the features used in the experiment = nfeatures
            path = working_dir + colname 
            pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
            pd.DataFrame(cols).to_csv(path + '/'  + 'features' + ".csv", header=['feature'], index=None)   
                         
            # returns a dictionary  
            diction[colname] = data
             
            #logger.info(df.label.unique())
            
           
        else:
                        
            target = df.columns.get_loc(colname)
            
            print()
            #logger.info(f'***** PLEASE NOTE THIS NUMBER {target} for {colname} *****')                       
            df_tr = df

            if not testset:
                X = df.iloc[:,0:-1].values
                Y = df.iloc[:,-1].values   
                i_X_train, i_X_test, i_y_train, i_y_test = train_test_split(X, Y, test_size = 0.4, random_state = 1)
                
                # save the test set
                data_test = pd.DataFrame(data = np.column_stack((i_X_test, i_y_test)), columns = dfcolumns)
                logger.success('Testset was separated from your dataset and will not be used in Data Implantation')
                data_test.to_csv('datasets/test_dataset.csv', index=False) 
                        
                df_tr = pd.DataFrame(data = np.column_stack((i_X_train, i_y_train)), columns = dfcolumns)
                
            new_colname = df_tr.columns[target]
            
            #create unique list of names
            unique_names = df_tr[new_colname].unique()
            
            # split the dateset into subset without preprocessing
            #create a data frame dictionary to store your data frames
            DataFrameDict = {elem : pd.DataFrame for elem in unique_names}
                      
            for key in DataFrameDict.keys():
                DataFrameDict[key] = df[:][df[colname] == key]
            
            diction = {}      
            for i, key in enumerate(DataFrameDict):
                # get the top level sub population male vs female
                df2 = DataFrameDict[key]
                identifier = colname+'_'+str(key)
                
                parent_dir = working_dir + colname + '/' + identifier
                data = []
                print()
                logger.info(f'Now creating {identifier} workspace ..... [Output artifacts will be stored here]')
                df2_copy = df2.copy()
                a = df2_copy.drop('label', axis=1)
                b = df2_copy.pop('label')
                
                for index in range(class_size):                       
                    # if total number of returned rows is greater than 0 
                   
                    if df2[df2['label']==index].shape[0] > 0:  
                        
                        # extract the focus_column
                        focus_column = df2[df2['label'] == index][colname]  
                        
                        if with_fs == True:
                            # pass X and y as parameters for feature selection
                            df_new, cols = self.doFeature_selection(a, b, nfeatures, colname)
                        else:
                            df_new, cols = df2, dfcolumns.tolist() 
                                                 
                        df3 = df_new[df_new['label'] == index].reset_index(drop=True)
                        
                        label = df3.iloc[:,-1] 
                        s_df = df3.drop('label', axis=1)
                         
                       
                        # directory = str(index)
                        # # save each sub-sub population into a seperate file
                        # path = os.path.join(parent_dir, directory)                         
                        # # Create the directory
                        # # 'index'
                        # try:
                        #     os.makedirs(path, exist_ok = True)
                        #     logger.info("Directory '%s' created successfully" % directory)
                        #     # if no of features is provided, make sure that no is greater than the number of PCA we want to do otherwise, use the number of PCA as teh no of features
                        #     if (nfeatures < no_pca):
                        #         nfeatures = no_pca                        
                        #     # doPCA and discard other data not needed and then save the PCA columns
                        #     # n_components must be between 0 and min(n_samples, n_features)
                        #     n_comp = nfeatures if min(s_df.shape[0], s_df.shape[1]) > nfeatures else min(s_df.shape[0], s_df.shape[1])
                        #     _, _, dictn, explained_variance = di.doPCA(s_df, n_comp)
                        #     ev_name = list(dictn.values())
                        #     evariance = list(explained_variance)
                        #     # merge the PCA info together for storate. Get first no_pca elements 
                        #     pd.DataFrame(np.column_stack((np.array(ev_name[0:no_pca]).flatten(), np.round(explained_variance[0:no_pca], 5)))).to_csv(path + '/evariance' + ".csv", header=['feature', 'ratio'], index=None)                     
                           
                        #     #save sub-sub population
                        #     pd.DataFrame(df3).to_csv(path + '/' + str(index) + ".csv", index=None)   
                        # except OSError as error:
                        #     logger.info("Directory '%s' can not be created" % directory)                                                
                        
                        # convert back to dataframe, where the last column is called label
                        if colname not in cols: # check if colname is in the list otherwise add it
                            s_df = np.column_stack((s_df, focus_column))
                            
                        s_data = np.column_stack((s_df, label))
                        
                        data.append(self.np_to_df(s_data))
                                       
                # to make sure that target column is part of the features being considered
                #if colname not in cols:
                    #evariance.append(0.0)
                    #ev_name.append(colname)
                # replace: discharge_mrs must always be part of the dataset --- its the label!!!                
                cols[cols.index('label')] = target_label
               
                # save the features used in the experiment = nfeatures
                path = working_dir + colname 
                
                pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
                pd.DataFrame(cols).to_csv(path + '/'  + 'features' + ".csv", header=['feature'], index=None)   
               
                target = cols.index(colname)
                
                if data: # list is not empty
                    diction[identifier] = data               
         
      
        return diction, target, unique_names, cols
    