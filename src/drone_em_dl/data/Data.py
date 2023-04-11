import pandas as pd
import numpy as np
from .data_util import *


class Data():
    def __init__(self):
        super(Data, self).__init__()
        self.callbacks = []
        
        
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass
    
    
    def load_data(self,
                  data:str='tobias.csv'):
        self.df = pd.read_csv(data)
        ''''
        self.df = pd.read_csv(data,sep=' ',header=None)
        if len(self.df.iloc[0,:])<2:
            
        if 'Time [ms]' not in self.df.columns:
            headers = ['time [s]', 
                        'Easting [m]', 
                        'Northing [m]', 
                        'relative flyvehøjde over jorde', 
                        'højden af jordes overflad',
                        'Heading [deg]', 
                        'Pitch [deg]', 
                        'Roll [deg]', 
                        'Estimeret 50Hz bidrag (parts-per-million)', 
                        'In-phase response 40025Hz',  
                        'Quadrature response 40025Hz', 
                        'In-phase response 65675Hz', 
                        'Quadrature response 65675Hz', 
                        'In-phase response 91275Hz', 
                        'Quadrature response 91275H']
            self.df.columns =headers
            del headers
        '''

        return None

    def get_features(self,input_idx:list = []):
        self.df_org = self.df
     
        if len(input_idx)>0:
            self.df = self.df.iloc[:,input_idx]

        return None

    def train_test_split(self,
                         split:float=0.8):
        

        
        self.org_train, self.org_test = train_test_split(self.df_org,split)
        self.train, self.test = train_test_split(self.df,split)

    def norm_data(self):
        self.norm_data = zscore_mean_man(self.train, self.test)


    def get_inv(self,data):
        data = inverse_zscore_mean_man(data,
                                       self.norm_data.mean_train,
                                       self.norm_data.std_train,
                                       self.norm_data.min_train,
                                       self.norm_data.max_train)
        return data



        
