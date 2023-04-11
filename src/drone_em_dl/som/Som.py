
class Som():
    '''

    
    '''
    
    def __init__(self):
        pass
        

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass
        
        
    def load_data(self,train,test):
        self.train_data = train
        self.test_data = test
    def plot_pc_explained_var(self,
                             explained_var:float=0.95):
        plot_explained_var(self.train_data,explained_var)

    def get_pca(self,pca_amount:int=9):
        self.pca_amount = pca_amount
        self.pca = PCA(pca_amount)
        self.pca.fit(self.train_data)

        self.pca_train = self.pca.transform(self.train_data)
        self.pca_test = self.pca.transform(self.test_data)

        self.pca_train_inv = self.pca.inverse_transform(self.pca_train)
        self.pca_test_inv = self.pca.inverse_transform(self.pca_test)

    def plot_pcs(self,test:bool=True):
        if test==True:
            plot_pcs(self.pca_test_inv,self.pca_amount,self.pca.explained_variance_ratio_)
        else:
            plot_pcs(self.pca_train_inv,self.pca_amount,self.pca.explained_variance_ratio_)


    

    
    

    
