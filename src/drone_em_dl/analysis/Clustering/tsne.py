
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import time




def get_tsne(data,n_components,verbose:int=1,**args):

    '''

    perplexity:
        a guess of the number of neighbours a point have. SNE should be fairly robust to the perplexity tho. 
        Somewhere between 5 and 50 is good according to the original paper.

        
    '''

    if verbose>0:
        time_start = time.time()
    tsne = TSNE(n_components=n_components, verbose=1, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    if verbose>0:
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    return tsne_results