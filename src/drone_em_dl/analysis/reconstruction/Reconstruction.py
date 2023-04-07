import numpy as np

class Reconstruction():
    '''
    fully connected ae for em data.
    
    '''
    
    def __init__(self):
        pass
        

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass
        
        
    def load_model(self,
                   model):
        self.model = model
        self.encoder = model.layers[1]
        self.decoder = model.layers[2]

    def get_reconstructions(self,data,keep_latent_varible:int=-1):

        self.latent_space = self.encoder.predict(data)
        if keep_latent_varible>-1:
            temp = np.zeros((self.latent_space.shape))
            temp[:,keep_latent_varible] = self.latent_space[:,keep_latent_varible]
            self.latent_space = temp
        decoded = self.decoder.predict(self.latent_space)
        self.reconstructions = decoded
        del decoded

        return None 

    def save_resutls(self,file_name:str=f'resuts'):
        np.save(f"{file_name}_reconstructions.npy",self.reconstructions)
        np.save(f"{file_name}_latent_space.npy",self.latent_space)
    

    
