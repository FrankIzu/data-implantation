#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 18:30:47 2022

@author: francis
"""

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import math
import time

from random import uniform
import random
from shapely.geometry import Polygon, Point, MultiPoint
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import scipy.spatial.distance as sdist
from scipy.spatial import ConvexHull, Delaunay
from sklearn.preprocessing import StandardScaler
from get_TSR_data_functions import TSR_data_functions


class TSR_DI_functions:       
    def implanFront(self, all_cluster):
      
        data = []
        cluster_poly_dict = {}
        nComp = 5 # number of components
        qtyIncr = 3 # extra excess points to be created since many will be rejected
        #print(len(all_cluster))
        
        # this code gets the number of samples belonging to each class. Then only those whose no of samples is greater than the no of components are considered
        # this is because PCA requires that the no of samples must be greater than the no of components
        list1 = []
        for c in all_cluster:
            list1.append(c[0].shape[0])
        list2 = [item for item in list1 if item > nComp]
            
        for index, cluster in enumerate(all_cluster):
            #print()
            print(f'running for cluster ... {cluster[2]}')          
            y_t = cluster[0]['label'] # cluster[0] is the dataset. [label] gives us the label
            y_t = pd.DataFrame(y_t, columns=['label'])
            
            temp = None
            temp_points = np.array([])
            #print(f'data shape is {cluster[1].shape} nComp is {nComp}')
            # since the number of samples MUST be greater than the number of components otherwise you we would get an error during PCA,
            # we have to check for that. If True, then we just accept those samples without implanting them
            if cluster[1].shape[0] < nComp: 
                # store in temp and add unto the next set
                temp = cluster[1]
                yhat = y_t['label'].values.tolist()                
                data.append(np.concatenate((temp, np.array([yhat]).T), axis=1))
                temp = None
                continue
            #if temp is not None:
                #data.append(temp)
                #temp = None
                                      
            X_train_pca, pca_model, dictn, ev = self.doPCA(cluster[1], nComp)
            dataset = pd.DataFrame(X_train_pca, columns = [f's{x}' for x in range(X_train_pca.shape[1])])  
            
            qty = math.floor(cluster[4]/len(list2)) # the quantity of new samples is divided amongst class with (enough) samples greater than the number of components
        
            # for implanting WITHIN the cluster, this value should be 1
            f = 1 # use this to control the distance multiplier. How far the points will be from each of the cluster
            
            accepted_points = np.array([])
            accepted_points.shape=(0, cluster[6]) # set A to be 0 rows x feature_size cols matrix
            accepted_pca = np.array([])
            accepted_pca.shape=(0, nComp)
            # loop until the required number of points that meet the criteria is reached
            #print(f'target quantity = {qty}')
                       
            while np.shape(accepted_points)[0] < qty:
                #ct+=1
                #print(f'miselenouse counter count = {ct}')
                # print(f'A2 = {np.shape(A2)[0]}, qty = {qty}') # this will show how points are created and stored
                A = new_poly = np.array([])
                A.shape=(0, nComp) # set A to be 0 rows x 2 cols matrix
                poly = X_train_pca         
               
                # determine the range for the new points
                min1, min2, min3, min4, min5  = min(X_train_pca[:, 0]), min(X_train_pca[:, 1]), min(X_train_pca[:, 2]), min(X_train_pca[:, 3]), min(X_train_pca[:, 4])
                max1, max2, max3, max4, max5  = max(X_train_pca[:, 0]), max(X_train_pca[:, 1]), max(X_train_pca[:, 2]), max(X_train_pca[:, 3]), max(X_train_pca[:, 4])                
           
               
                #print(f'{min1} {min2} {min3} {min4} {min5}')                
                #print(f'{max1} {max2} {max3} {max4} {max5}')
                break_out_flag = False
                try:
                    # create qty*qtyIncr number of new points
                    points = np.column_stack((np.random.randint(min1*f, max1*f, size=qty*qtyIncr), np.random.randint(min2*f, max2*f, size=qty*qtyIncr), np.random.randint(min3*f, max3*f, size=qty*qtyIncr), np.random.randint(min4*f, max4*f, size=qty*qtyIncr), np.random.randint(min5*f, max5*f, size=qty*qtyIncr)))                                  
                    
                except ValueError:
                    break_out_flag = True
                    break
                
                #print (f'{points.shape} created')
                # check if the new points are inside the new polygon. To check for outside the polygon use < or False
                A = points[list(*np.where((Delaunay(poly).find_simplex(points) >= 0)==True)), :]
                #A = np.array(self.in_poly_hull_multi(poly, points)) # ---- this works but much slower than Delaunay             
                #print (f'{A.shape} are inside')
                # check if the new points are outside the all the other polygons.  
                if index > 0:
                    #print(f'there are {len(cluster_poly_dict)} keys')
                    for poly_value in cluster_poly_dict.values():
                        #print(f'poly_value is {poly_value.shape}')   
                        #print(f'A before is {A.shape}')
                        # checking for outside the polygon
                        try:
                            A = A[list(*np.where((Delaunay(poly_value).find_simplex(A) >= 0)==False)), :] # test for outside OTHER polygon
                        except:
                            #print('QError was thrown but execution was continued.') # QhullError
                            #print(f'shape of the polygon was {poly_value.shape}')
                            #print(f'shape of the point was {A.shape}')
                            
                            # when there is an error, PASS: accept the points anyhow OR CONTINUE: neglect that polygon check OR BREAK: discard all the points
                            break
                        
                            # creating new points when a QhullErrro occurs does not help becasue the error is in the polygon not on the points 
                            #A = np.random.rand(1, nComp) # remove conflicted data and reinitialize                             
                        #print(f'A after is {A.shape}')
                        if A.shape[0] < nComp: # ensure that 0 points or conflicted data is not added to dictionary
                            continue
                        else:
                            #print(f'added to dict with key {cluster[2]}')
                            new_poly = A       
                            #print (f'{A.shape} made it polygon wide')
                else:
                    # the first cluster that has nothing to compare with
                    new_poly = A                   
                               
                
                # convert back to dataframe
                ndf = pd.DataFrame(new_poly, columns = [f's{x}' for x in range(new_poly.shape[1])])                                        
                mu = np.mean(cluster[1].values.astype(int), axis=0)
                nXhat = np.dot(ndf, pca_model.components_[:nComp,:])
                nXhat += mu
                
                #print(sdist.euclidean(nXhat[0,:], centroids[0,:]))
                
                # farthest point from each cluster
                farthest_point = cluster[0].loc[cluster[0]['Cluster']==cluster[2]]['dist_'+str(cluster[2])].max()
               
                accepted_points_count=0
                for loc, row in enumerate(nXhat):
                   # distance of each point from the centroid
                   dist_from_cent = sdist.euclidean(row, cluster[7][cluster[2],:])
                   if dist_from_cent < farthest_point: # use the < sign if you want points within the boundry
                       accepted_points = np.concatenate((accepted_points, np.reshape(np.array(row), (1, -1)))) # convert to 2D and concate                      
                       #print(f'index = {index} and new_poly = {new_poly[index,:]}')
                       accepted_pca = np.concatenate((accepted_pca, np.reshape(np.array(new_poly[loc,:]), (1, -1))))
                       accepted_points_count=accepted_points_count+1
                
                #print(f'We want = {qty}: {accepted_points_count} added --> total is now {accepted_points.shape[0]}')                
                cluster_poly_dict[cluster[2]] = accepted_pca
                #for key, value in cluster_poly_dict.items():
                    #print(key)
            
            if break_out_flag == True:
                continue
            
            # create extra y values to correspond wth the extra x data points created          
            yhat = y_t['label'].values.tolist() + [cluster[3]]*(qty)
            #print(f'the label is {[cluster[3]]*(qty)}')
               
            # take only the number of points we want (qty) and discard the rest            
            qXhat = accepted_points[0:qty,:]
            #print(f'{accepted_points.shape}. qty is {qty} qxHat is {qXhat.shape}')
            nX_train = np.concatenate((cluster[1], qXhat))
            
            ny_train = yhat 
                  
            #print(f'Original {ds2.shape}')
            #print (f'number of Implanted points = {qXhat.shape}')
            
            #print(f'for x {nX_train.shape}')
            #print(f'for y {np.array([ny_train]).T.shape}')
            
            data.append(np.concatenate((nX_train, np.array([ny_train]).T), axis=1))
            #pd.DataFrame(data).to_csv("datasets/research/BCW_"+ name +"_implanted.csv", index=None)
        
        return  data
        
    def validate_cluster_by_centroid(self, all_cluster_data, class_num, feature_size, current):
        # the points here have been validted against all clusters and found to be legitimately wihtin their cluster
      
        for index, c_tuple in enumerate(all_cluster_data):
            if index == current:
                continue
            accepted_points = np.array([])
            accepted_points.shape=(0, feature_size) # set A to be 0 rows x feature_size cols matrix           
            accepted_points_count = 0
            print(f'cluster {current} has {all_cluster_data[current][0].shape} and tested against {c_tuple[0].shape} ')
            
            for sample in all_cluster_data[current][0][:,0:-1]: # gives us all rows and all features without the label for a given cluster
                # check if any sample falls within another cluster
                dist_from_cent = sdist.euclidean(sample, c_tuple[2][c_tuple[1],:])
                if (dist_from_cent > c_tuple[3]): # use > to make sure the sample is outside
                    accepted_points = np.concatenate((accepted_points, np.reshape(np.array(sample), (1, -1))))
                    accepted_points_count=accepted_points_count+1
            
            # rebuilding the dataset by adding back the y label values after it was removed
            yhat = [str(class_num)] * (len(accepted_points))
            accepted_points = np.column_stack((accepted_points, yhat))
          
            print(f'cluster {index} now has {accepted_points.shape} records')  
            if accepted_points.shape[0] < all_cluster_data[current][0].shape[0]:
                print(f'{all_cluster_data[current][0].shape[0] - accepted_points.shape[0]} conflicts detected !!!')
            #print ('**********************************')
        
        # replace the bad data with the good data
        all_cluster_data[current][0] = accepted_points
        
        return all_cluster_data
        
    def validate_cluster_by_point_check(self, all_cluster_data, class_num, feature_size, current):
        for index, c_tuple in enumerate(all_cluster_data):
           if index == current:
               continue
           accepted_points = np.array([])
           accepted_points.shape=(0, feature_size) # set A to be 0 rows x feature_size cols matrix           
           accepted_points_count = 0
           print(f'cluster {current} has {all_cluster_data[current][0].shape} and tested against {c_tuple[0].shape} ')
           for sample in all_cluster_data[current][0][:,0:-1]: # gives us all rows and all features without the label for a given cluster                   
                # check if any sample falls within another polygon
                poly = c_tuple[0][:,0:-1]
                points = sample
                #accepted_points = points[list(*np.where((Delaunay(poly).find_simplex(points) >= 0)==True)), :]
                accepted_points = self.in_poly_hull_multi(poly, points)
                accepted_points = np.concatenate((accepted_points, np.reshape(np.array(sample), (1, -1))))
                accepted_points_count=accepted_points_count+1
                print(accepted_points_count)
                
           # rebuilding the dataset by adding back the y label values after it was removed
           yhat = [str(class_num)] * (len(accepted_points))
           accepted_points = np.column_stack((accepted_points, yhat))
         
           #print(f'cluster {index} now has {accepted_points.shape} records')  
           if accepted_points.shape[0] < all_cluster_data[current][0].shape[0]:
               print(f' Anomaly detected !!!')
           #print ('**********************************')
        
        # replace the bad data with the good data
        all_cluster_data[current][0] = accepted_points
        
        return all_cluster_data
        
    def doPCA(self, df_tr, nComp):
            
        #normalize data
        Standardized_data = StandardScaler().fit_transform(df_tr)
        df_tr =  TSR_data_functions.np_to_df(Standardized_data)        
        
        #print(df_tr.shape)
        if df_tr.shape[0] < nComp:
            nComp = df_tr.shape[0]
        pca_model = PCA(n_components=nComp).fit(df_tr)
        X_train_pca = pca_model.transform(df_tr)
        
        # number of components
        n_pcs= pca_model.components_.shape[0]
        
        # get the index of the most important feature on EACH component i.e. largest absolute value
        most_important = [np.abs(pca_model.components_[i]).argmax() for i in range(n_pcs)]
        
        initial_feature_names = [col_name for col_name in df_tr.columns] #[f's{x+1}' for x in range(df_tr.shape[1])]
        
        # get the names
        most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
        
        # using LIST COMPREHENSION HERE AGAIN
        dictn = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}
        
        # build the dataframe
        df = pd.DataFrame(sorted(dictn.items()))
        
        explained_variance = pca_model.explained_variance_ratio_
            
        return X_train_pca, pca_model, dictn, explained_variance
    
    def implantFiFl(self, d, name, dataset, size2, nComp):
                
        #d = df2    
           
        ds = d.drop('label', axis=1)
        
        # we want to increase the size of the minority samples by x%
        p = round(int(round((len(ds)/len(dataset))*size2)))
        
        #  # generate random 3D points
        allCols = np.array([])
        for x in range(1, nComp+1):
            uCol = [uniform(d['Column'+str(x)].min(), d['Column'+str(x)].max()) for i in range(0, p)]
            if x==1:
                allCols = np.concatenate([allCols, uCol])
            else:
                allCols = np.column_stack((allCols, uCol))                
                        
        return allCols, len(allCols)
   
    def in_poly_hull_multi(self,  poly, points):
        hull = ConvexHull(poly)
        res = []
        for p in points:
            new_hull = ConvexHull(np.concatenate((poly, [p])))
            #res.append(np.array_equal(new_hull.vertices, hull.vertices))
            if not np.array_equal(new_hull.vertices, hull.vertices): res.append(p)
        return res

