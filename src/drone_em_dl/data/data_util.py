import numpy as np
from collections import namedtuple
import pandas as pd


def train_test_split(df,
                     split:float=0.8):
    df = df.sample(frac=1,random_state=42)
    train= df.sample(frac=split,random_state=42) 
    test= df.drop(train.index)
    return train,test





def minmax(data,minimum,maximum):
    return (data-minimum)/(maximum-minimum)


def inv_minmax(data,minimum,maximum):
    return data*(maximum-minimum)+ minimum


def int_zscore(data,mean,std):
    return data*std+mean

def zscore(data,mean,std):
    return (data-mean)/std


def zscore_mean_man(train,test):
    Normlised_data = namedtuple('Normlised_data', ['norm_train', 'norm_test', 'mean_train','std_train','min_train','max_train','zscore_train','zscore_test'])

    try:
        train = train.to_numpy()
        test = test.to_numpy()
    except:
        pass

    train_mean = np.mean(train,axis=0)
    train_std = np.std(train,axis=0)
    
    zscore_train = zscore(train,train_mean,train_std)
    zscore_test = zscore(test,train_mean,train_std)
    
    
    train_min = np.min(zscore_train,axis=0)
    train_max = np.max(zscore_train,axis=0)

    norm_train = minmax(zscore_train,train_min,train_max)
    norm_test = minmax(zscore_test,train_min,train_max)

    return Normlised_data(norm_train, norm_test,train_mean, train_std, train_min, train_max,zscore_train,zscore_test)






def inverse_zscore_mean_man(data,mean,std,minimum,maximum):


    data = inv_minmax(data,minimum,maximum)
    data = int_zscore(data,mean,std)

    return data 





