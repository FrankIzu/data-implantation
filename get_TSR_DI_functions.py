#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:00:58 2021

@author: francis
"""

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import math

from random import uniform
import random
from shapely.geometry import Polygon, Point, MultiPoint
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import scipy.spatial.distance as sdist
from scipy.spatial import ConvexHull, Delaunay
                      
#https://stackoverflow.com/questions/29311682/finding-if-point-is-in-3d-poly-in-python

class TSR_DI_functions:       
    def implanFront(self, ds1, ds2, c, name, class_num, need, no_of_clusters, feature_size, centroids, class_size):
        
        nComp = 3               
        y_t = ds1['label']
        y_t = pd.DataFrame(y_t, columns=['label'])
         
        # #newX_train = data1.iloc[:,:-1].values.astype(int)
        
        # #Y = dff1["label"]
        # labelencoder_y = LabelEncoder()
        # Y = labelencoder_y.fit_transform(y_t)
        # y_t = pd.DataFrame({'label': Y})
            
    # =============================================================================
    #     nComp = 3 # 4 is also good
    #     pca = PCA(n_components=nComp)
    #     X_train_pca = pca.fit_transform(ds2)
    #     explained_variance = pca.explained_variance_ratio_
    # =============================================================================
        
        X_train_pca, pca_model, dictn, ev = self.doPCA(ds2, nComp)
        
        dataset = pd.DataFrame({'Column1': X_train_pca[:, 0], 'Column2': X_train_pca[:, 1], 'Column3': X_train_pca[:, 2]})
        
        #print(f'dataset {dataset.shape}')
               
        qty = math.floor(need/no_of_clusters)
        #print(f'To be created = {qty}')
        
        f = 1 # use this to control the distance multiplier. How far the points will be from eacch the cluster
        
        # generate random 3D points and test if points are inside polygon
        def get_random_point_in_polygon(poly):
             minx, miny, minz, maxx, maxy, maxz = min(X_train_pca[:, 0]), min(X_train_pca[:, 1]), min(X_train_pca[:, 2]), max(X_train_pca[:, 0]), max(X_train_pca[:, 1]), max(X_train_pca[:, 2]) #poly.bounds
             while True:
                 
                 p = Point(random.uniform(minx*f, maxx*f), random.uniform(miny*f, maxy*f), random.uniform(minz*f, maxz*f))         
                 if poly.contains(p) == True: # toggle between TRUE and FALSE. True will generate the points within the region
                     return p
                        
        A = np.array([])
        A.shape=(0,3) # set A to be 0 rows x 2 cols matrix
        for x in range(qty*2):   # instead of doing this (qty*3), we can create a seperate funcction to generate points and call it 
                        # several times until A2 has the desired number of points. In that case, we will append to what A2 has each time
           
            # either of the below polygon will work
            # p = Polygon(X_train_pca) 
            p = MultiPoint(X_train_pca).convex_hull
            point_outside_poly = get_random_point_in_polygon(p)
            array = np.array([[(point_outside_poly.x), (point_outside_poly.y), (point_outside_poly.z)]])
            A = np.concatenate((A, array))
                
        
        # convert to dataframe the pca points that are outside the polygon
        ndf = pd.DataFrame(
            {'Column1': A[:,0],
             'Column2': A[:,1],
             'Column3': A[:,2],
            })
        
        #df = pd.concat([dataset, uDf, lDf])
        
        mu = np.mean(ds2.values.astype(int), axis=0)
        nXhat = np.dot(ndf, pca_model.components_[:nComp,:])
        nXhat += mu
        
        #print(sdist.euclidean(nXhat[0,:], centroids[0,:]))
        
        # farthest point from each cluster
        c0 = ds1.loc[ds1['Cluster']==c]['dist_'+str(c)].max()
        #c1 = dff1.loc[dff1['Cluster']==1]['dist_1'].max()
        #c2 = dfX.loc[dfX['cluster']==2]['dist_2'].max()
        
        A2 = np.array([])
        A2.shape=(0, feature_size) # set A to be 0 rows x 2 cols matrix
        ct=0
        for row in nXhat:
           #if(row['cluster'] = 1)
           d0 = sdist.euclidean(row, centroids[c,:])
           #d1 = sdist.euclidean(row, centroids[1,:])
           #d2 = sdist.euclidean(row, centroids[2,:])
           if d0<c0: # use the < sign if you want points within the cluster
               A2 = np.concatenate((A2, np.reshape(np.array(row), (1, -1))))
               ct=ct+1
        
        #print(ct)
               
            
        # create extra y values to correspond wth the extra x data points created    
        yhat = y_t['label'].values.tolist() + [class_num]*(qty)
        #print(f'the label is {[class_num]*(qty)}')
           
           
        qXhat = A2[0:qty,:]
        nX_train = np.concatenate((ds2, qXhat))
        ny_train = yhat 
          
        #print(f'Original {ds2.shape}')
        #print (f'number of Implanted points = {qXhat.shape}')
        
        #print(f'for x {nX_train.shape}')
        #print(f'for y {np.array([ny_train]).T.shape}')
        data = np.concatenate((nX_train, np.array([ny_train]).T), axis=1)
        #pd.DataFrame(data).to_csv("datasets/research/BCW_"+ name +"_implanted.csv", index=None)
        
        return data
    
    def doPCA(self, df_tr, nComp):
            
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
   
    def in_poly_hull_multi(poly, points):
        hull = ConvexHull(poly)
        res = []
        for p in points:
            new_hull = ConvexHull(np.concatenate((poly, [p])))
            #res.append(np.array_equal(new_hull.vertices, hull.vertices))
            if np.array_equal(new_hull.vertices, hull.vertices): res.append(p)
        return res