class Fae():
    '''
    fully connected ae for em data.
    
    '''
    
    def __init__(self):
        pass
        

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass
        
        
        
        
    def make_model(self,
                   input_size:tuple=(7,),
                   latent_space_dim:int=5,
                   dropout_prob:float=0.2,dense_neurons:list = [228,128,56],name:str="AE_tobias"):
        
        
        
        
        # Encoder
        
        x = tf.keras.layers.Input(shape=input_size, name="encoder_input")
        encoder = x
        encoder2 = x
        
        initializer = tf.keras.initializers.LecunNormal()
        
        for i,neurons in enumerate(dense_neurons):
            encoder = tf.keras.layers.Dense(neurons, name=f"encoder_dense_{i+1}",activation='selu',kernel_initializer=initializer)(encoder)
            encoder = tf.keras.layers.BatchNormalization(name=f"encoder_norm_{i+1}")(encoder)
            #encoder = tf.keras.layers.selu(name=f"encoder_leakyrelu_{i+1}")(encoder)
            encoder = tf.keras.layers.Dropout(dropout_prob,name=f"encoder_dropout_{i+1}")(encoder)

        shape_before_flatten = tf.keras.backend.int_shape(encoder)[1:]
        encoder_flatten = tf.keras.layers.Flatten()(encoder)
        encoder_output = tf.keras.layers.Dense(units=latent_space_dim, name="encoder_output")(encoder_flatten)
        
        
        for i,neurons in enumerate(dense_neurons):
            encoder2 = tf.keras.layers.Dense(neurons, name=f"encoder2_dense_{i+1}",activation='selu',kernel_initializer=initializer)(encoder2)
            encoder2 = tf.keras.layers.BatchNormalization(name=f"encoder2_norm_{i+1}")(encoder2)
            #encoder = tf.keras.layers.selu(name=f"encoder_leakyrelu_{i+1}")(encoder)
            encoder2 = tf.keras.layers.Dropout(dropout_prob,name=f"encoder2_dropout_{i+1}")(encoder2)




        encoder2 = tf.keras.layers.Flatten()(encoder2)
        encoder2 = tf.keras.layers.Dense(units=latent_space_dim, name="encoder2_output")(encoder2)
        
        
        encoder12_output = tf.keras.layers.Add()([encoder_output, encoder2])
        
        encoder_model = tf.keras.models.Model(x, encoder12_output, name="encoder_model")
        # Decoder

        
        

        
        
        
        decoder_input = tf.keras.layers.Input(shape=(encoder_output.shape[1:]), name="decoder_input")
        decoder = tf.keras.layers.Flatten()(decoder_input)
        dense_neurons.reverse()
        for i,neurons in enumerate(dense_neurons):
            decoder = tf.keras.layers.Dense(neurons, name=f"decoder_dense_{i+1}",activation='selu',kernel_initializer=initializer)(decoder)
            decoder = tf.keras.layers.BatchNormalization(name=f"decoder_norm_{i+1}")(decoder)
            #decoder = tf.keras.layers.selu(name=f"decoder_leakyrelu_{i+1}")(decoder)
            decoder = tf.keras.layers.Dropout(dropout_prob,name=f"decoder_dropout_{i+1}")(decoder)
        
        
        decoder = tf.keras.layers.Dense(input_size[-1], name=f"decoder_output",activation=None)(decoder)
        decoder = tf.keras.layers.Reshape(input_size)(decoder)
        
        decoder_model = tf.keras.models.Model(decoder_input, decoder, name="decoder_model")
        

        ae_input = tf.keras.layers.Input(shape=input_size, name="AE_input")
        ae_encoder_output = encoder_model(ae_input)
        ae_decoder_output = decoder_model(ae_encoder_output)



        ae = tf.keras.models.Model(ae_input, ae_decoder_output, name=name)

        optimizer = optimizer=tf.keras.optimizers.Adam(amsgrad=True,
                                                      clipnorm=1, 
                                                      clipvalue=1.0,learning_rate=0.01)


        lr_metric = get_lr_metric(optimizer)
        
        ae.compile(optimizer, 
                   loss=tf.keras.losses.MeanSquaredError(), 
                   metrics=['mse',lr_metric])

        return ae 