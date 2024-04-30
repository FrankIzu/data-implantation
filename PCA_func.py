import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
    
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

import numpy as np
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import scipy.spatial.distance as sdist
from numpy import unique
from matplotlib import pyplot
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns #visualisation library
from sklearn.decomposition import PCA

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import math

from random import uniform
import random
from shapely.geometry import Polygon, Point, LinearRing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import scipy.spatial.distance as sdist

import get_TSR_data_functions as tsr_d
import get_TSR_DI_functions as tsr_f




import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)




df = pd.read_csv("datasets/wholeset_Jim_nomissing_validated.csv")    
    
df = df.drop(['IDCASE_ID','ICASE_ID','MRS_1','MRS_3'], axis=1)    
  
col = df.pop("discharged_mrs")
df.insert(df.shape[1], col.name, col)

df = df.rename(columns = {'discharged_mrs':'label'}) 

X = df.iloc[:,0:-1].values
Y = df.iloc[:,-1].values   

pca_model = PCA(n_components=3).fit(X)

X_train_pca = pca_model.transform(X)

# number of components
n_pcs= pca_model.components_.shape[0]

# get the index of the most important feature on EACH component i.e. largest absolute value
most_important = [np.abs(pca_model.components_[i]).argmax() for i in range(n_pcs)]

initial_feature_names = [col_name for col_name in df.columns] #[f's{x+1}' for x in range(df_tr.shape[1])]

# get the names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

# using LIST COMPREHENSION HERE AGAIN
dictn = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}

# build the dataframe
df2 = pd.DataFrame(dictn.items())

explained_variance = pca_model.explained_variance_ratio_







