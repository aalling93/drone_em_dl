from sklearn.cluster import DBSCAN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors



def find_epsilon(data,k=2):
    ''''
    the maximum distance at which another observation must be to be considered to meet the criterion of “being close”.. How to pick.

    we are using knn 

    '''

    nearest_neighbors = NearestNeighbors(n_neighbors=k)
    neighbors = nearest_neighbors.fit(data)
    distances, indices = neighbors.kneighbors(data)
    distances = np.sort(distances, axis=0)

    distances = distances[:,1]

    return distances, indices


def get_dbscan(data,eps:float=0.25,min_sample:int=2,**args):
    dbscan = DBSCAN(eps=eps, min_samples=min_sample)
    dbscan.fit(data)
    labels = dbscan.labels_

    return dbscan,labels