# Support Vector Machine (SVM)

# Importing the libraries

import os 

import numpy as np
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
from sklearn.metrics import silhouette_score

from statistics import mode

import scipy.spatial.distance as sdist
from numpy import unique
from matplotlib import pyplot
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns #visualisation library
from sklearn.decomposition import PCA
import warnings

from .get_TSR_data_functions import TSR_data_functions
from .get_TSR_DI_functions_multiclass import TSR_DI_functions

warnings.filterwarnings("ignore")
print('*** modification tracking - v18 ***')

class DataImplantation_Frontier:
        
    def __init__(self, col = "NoTarget", nF=-1, createtest=False):        
        self.colname = col
        self.nfeatures = nF
        self.createtest = createtest
    
    def runDIprocess(self, dataset, labelcolumnname = "label", class_size = 2, di_ratio=100, doDI=True):
        percentage_of_outlier = di_ratio # value is in percentage. set to 0% to bypass DI
        #self.colname = 'GENDER_TX'
        
        working_dir = os.getcwd() + "/datasets/"
        
        # call participating classes
        func = TSR_data_functions()
        di = TSR_DI_functions()
        
        diction, target, unique_names, cols = func.get_TSR_by_Class_generic(dataset, class_size, self.colname, self.nfeatures, labelcolumnname, self.createtest)
       
        category_list = []
        #d=  func.get_TSR_by_Class('CA_ID')
        
        for key, value in diction.items():
            category = key
            data_dictn = value
            
            print(); # empty line
            if doDI==True:
                print(f'Implantation for {category} started ......')
            else:
                 print(f'Sub {category} contains categorocal data. No DI will be performed.')
        
            # get the biggest dataframe
            #max_size = max([data_dictn[0].shape[0], data_dictn[1].shape[0], data_dictn[2].shape[0], data_dictn[3].shape[0], data_dictn[4].shape[0], data_dictn[5].shape[0]])
            max_size = 0
            kmeans_k = []   
            for i in range(len(data_dictn)):
                if data_dictn[i].shape[0] > max_size:
                    max_size = data_dictn[i].shape[0] 
                    
                # determine the best number of clusters for the algorithm
                #kmeans_k.append(func.get_no_of_clusters_dbscan(data_dictn[i]))
                kmeans_k.append(func.get_no_of_clusters_kmeans(data_dictn[i]))
            no_of_clusters = mode(kmeans_k)
            #print(f'no of clusters = {no_of_clusters}')
                            
            feature_size = len(data_dictn[0].columns) - 1 # exclude the label
            
            appended_data = [] #np.array([[]])
                 
                                
            for class_num in range(0, len(data_dictn)):
                all_cluster = []     
                mdata = pd.DataFrame()
                #mdata = np.array([])
                #mdata.shape=(0, feature_size + 1) # set mdata to be 0 rows x feature_size cols matrix. +1 is the label
                                
                data0= data_dictn[class_num]
                    
                # we want to creat 100% outlier - 1:1 ratio
                goal = int(round((percentage_of_outlier/100)*max_size))
                
                # we already have some minority samples   
                need = goal-len(data0)
                #need_ratio = need/len(data0)
                                            
                #print(f'... reducing the size of points ***** this is temporary {need-1000}')
                #need = need/2
                
                if need > 2 and doDI==True:    
                    print()
                    print(f'Now implanting ... label {class_num}')
                    
                    print(f'Largest = {max_size} Goal = {goal}')
                    print(f'what we have = {len(data0)} what we need to create ~ {goal-len(data0)}')
                        
                    dfX = data0.copy()
                    
                    #Y = dfX["label"]
                    points = dfX.drop('label', axis=1)   
                                       
                    # or points = df[['Type1', 'Type2', 'Type3']]
                    #print(f'maximum = {points.max()}')
                    kmeans = cluster.KMeans(n_clusters=no_of_clusters, random_state=0).fit(points)
                    dfX['Cluster'] = kmeans.labels_
                    #dfX = pd.concat([dfX, Y], axis=1)
                    
                    # get the distance between each sample from the centroids
                    centroids = kmeans.cluster_centers_
                    dists = pd.DataFrame(
                        sdist.cdist(points, centroids), 
                        columns=['dist_{}'.format(i) for i in range(len(centroids))],
                        index=dfX.index)
                    # print(dists);
                    dfX = pd.concat([dfX, dists], axis=1)
                    
                    dfXX = dfX.copy()
                    
                    for i in range(no_of_clusters):
                        dff = dfXX[dfXX['Cluster'] == i]
                        dc = dff.drop('label', axis=1).drop('Cluster', axis=1)                            
                        for j in range(no_of_clusters):
                            dc = dc.drop('dist_'+str(j), axis=1)
                        all_cluster.append([dff, dc, i, class_num, need, no_of_clusters, feature_size, centroids])
                    
                    for data in di.implanFront(all_cluster):
                        mdata = pd.concat([mdata, pd.DataFrame(data = data, columns = cols)], axis=0)
                             
                    
                    # perform extra validation to make sure that points within the clusters DO NOT fall into any other cluster
                   
                    # for i in range(no_of_clusters):  
                    #     all_cluster_data = di.validate_cluster_by_point_check(all_cluster_data, class_num, feature_size, i)
                    #     print()
                                           
                else:
                    mdata = pd.DataFrame(data = data0.to_numpy(), columns = cols)                    
                    
                #print(f'Before implant {data0.shape}')
                print(f'After implant {mdata.shape} *** Goal was {(goal, mdata.shape[1])}')
                appended_data.append(mdata) 
                
            merged_dataset = pd.concat(appended_data)
            #np.random.shuffle(merged_dataset)
            print()
                                               
            print(f'Total samples merged = {merged_dataset.shape}')
            
            #np.unique(y_train, return_counts=True)     
            if self.colname != 'NoTarget':             
                pd.DataFrame(merged_dataset).to_csv(working_dir + self.colname + '/' + category + '/TSR_' + category +"Front_implanted.csv", index=None)
        
        parent_dir = working_dir + self.colname + '/TSR_'
        if self.colname == 'NoTarget':   
            pd.DataFrame(merged_dataset).to_csv(parent_dir + self.colname + '_' + "Front_implanted.csv", index=None)
        else:       
            pd.DataFrame(merged_dataset).to_csv(parent_dir + self.colname + '_' + "Front_implanted.csv", index=None)
                        
        return merged_dataset, target, unique_names
    
# TEST CASES

# passing -1 as feature number means feature selection will not be used 
# passing any numbers less than 3 will hurt the program. Feature selection needs more than 2 features to proceed.
#dataset = pd.read_csv("datasets/cleaned.csv") #.drop(['HOSP_MRS90_DATE'], axis=1)
#process = DataImplantation_Frontier() # provide name of your label column; Do you want to separate your testset?
#process.runDIprocess(dataset) # class size; if we should DataImplant or not. % of implant to equalize the dataset

#process = DI_Frontier_TSR('HIST_CEREB_ISCH', -1) #GCSE_NM
#target, unique_names = process.runDIprocess(7, True)

# Don't be disappointed if Data Implantation does not give you as much synthetic samples as you wanted.
# DI uses advanced technique to generate samples that are as close to real samples as possible; 
# it will not generate outliers in the name of give you the desired quantity of samples
# To get more samples, you can try increasing the pecentage of synthetic samples to a number greater than 100%