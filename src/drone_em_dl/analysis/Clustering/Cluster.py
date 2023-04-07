import numpy as np
from .dbscan import *
from .tsne import *

class Cluster():
    def __init__(self):
        super(Cluster, self).__init__()
        self.callbacks = []
        
        
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass
    
    
    def load_data(self,
                  data:np.ndarray):
        self.data = data 

    def epsilon(self,neighbours:int=2,**args):
        distances, indices = find_epsilon(self.data,k=neighbours,**args)
        return distances, indices

    def dbscan(self,epsilon:float=0.19,min_sample:int=2,**args):
        self.dbscan_clusters, self.label = get_dbscan(self.data,epsilon,min_sample=min_sample,**args)

    def tsne(self,dimensions:int=2,perplexity:float=30,**args):
        self.tsne_clusters = get_tsne(self.data,dimensions,perplexity=perplexity,**args)



        