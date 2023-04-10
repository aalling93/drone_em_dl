import tensorflow.compat.v2 as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd
import os
from .fae import Fae


def model_builder(hp):
    """
    Build model for hyperparameters tuning

    hp: HyperParameters class instance
    """
    tf.keras.backend.clear_session()
    # defining a set of hyperparametrs for tuning and a range of values for each
    # filters = hp.Int(name = 'filters', min_value = 60, max_value = 230, step = 20)
    filterSize = hp.Int(name="filterSize", min_value=2, max_value=7, step=1)
    latent_space_dim = hp.Int(name="latentSpaceDim", min_value=1, max_value=20, step=1)

    clip = hp.Float("clipping", min_value=0.4, max_value=1, step=0.2)
    learning_rate = hp.Float("lr", min_value=1e-6, max_value=1e-2, sampling="log")

    filters = []
    for i in range(filterSize):
        filters.append(
            hp.Int(name=f"filters{i+1}", min_value=60, max_value=500, step=50)
        )

    drop_rate = hp.Float(name="dropout_prob", min_value=0, max_value=0.4, step=0.05)
    laten_space_regularisation_L1 = hp.Float(
        name="laten_space_regularisation_L1",
        min_value=0.00001,
        max_value=0.001,
        step=0.05,
    )

    with Fae() as ae:
        ae = ae.make_model(
            input_size=(25,),
            latent_space_dim=latent_space_dim,
            dense_neurons=filters,
            dropout_prob=drop_rate,
            laten_space_regularisation_L1=laten_space_regularisation_L1,
            name="AE_hyperparamters",
        )

    optimizer = optimizer = tf.keras.optimizers.Adam(
        clipnorm=clip, clipvalue=clip, learning_rate=learning_rate
    )
    lr_metric = get_lr_metric(optimizer)
    ae.compile(
        optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=["mse", lr_metric]
    )

    return ae


def load_gpu(which: int = 0, memory: int = 60000):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(which)
    print("loading gpu")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(gpus)
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)],
        )
    except RuntimeError as e:
        print("\n error '\n")
        print(e)
    strategy = tf.device(f"/GPU:0")

    return strategy


def get_lr_metric(optimizer):
    @tf.autograph.experimental.do_not_convert
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


def get_callbacks(model):
    """ """
    os.makedirs(f"{model.name}", exist_ok=True)
    best_model_file = f"{model.name}/best_model_{model.name}.h5"
    early_stop = EarlyStopping(monitor="val_loss", patience=100)
    best_model = ModelCheckpoint(
        best_model_file, monitor="val_loss", mode="auto", verbose=0, save_best_only=True
    )
    log_dir = f"logs/{model.name}"

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq="epoch",
        profile_batch=2,
        embeddings_freq=1,
    )

    cbk = CustomModelCheckpoint(model)
    reduce_lr = ReduceLROnPlateau(
        monitor="loss", factor=0.002, patience=20, min_lr=1e-25
    )
    callbacks = [best_model, early_stop, reduce_lr, cbk, tensorboard_callback]

    return callbacks


def model_fit(
    model,
    X_train,
    X_val,
    batch_size: int = 200,
    epochs: int = 300,
    verbose: int = 1,
    callbacks: list = [],
):
    training_history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        verbose=verbose,
        shuffle=True,
        batch_size=batch_size,
        callbacks=callbacks,
    )


class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super().__init__()
        self.epoch_count = 0
        self.learning_rates = []
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0:
            if self.model.history.history["val_loss"][-1] == min(
                self.model.history.history["val_loss"]
            ):
                self.model.save_weights(
                    f"models/{self.model.name}/weights_lowest_loss_{self.model.name}.h5",
                    overwrite=True,
                )
                pd.DataFrame(self.model.history.history).to_pickle(
                    f"models/{self.model.name}/history_lowest_loss_epoch_{self.model.name}.pkl"
                )

            lr = K.get_value(self.model.optimizer.lr)
            self.learning_rates.append(lr)
            self.model.save(
                f"models/{self.model.name}/model_{self.model.name}_newest_epoch.h5",
                overwrite=True,
            )
            # self.model.save(f'{self.model.name}/model_{self.model.name}_newest_epoch', overwrite=True)

            self.model.save_weights(
                f"models/{self.model.name}/weights_{self.model.name}_newest_epoch.h5",
                overwrite=True,
            )
            pd.DataFrame(self.model.history.history).to_pickle(
                f"models/{self.model.name}/history_newest_epoch_{self.model.name}.pkl"
            )

        if epoch % 100 == 0:
            self.model.save(
                f"models/{self.model.name}/model_{self.model.name}_epoch_{epoch}.h5",
                overwrite=True,
            )


class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super().__init__()
        self.epoch_count = 0
        self.learning_rates = []
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0:
            if self.model.history.history["val_loss"][-1] == min(
                self.model.history.history["val_loss"]
            ):
                self.model.save_weights(
                    f"{self.model.name}/weights_lowest_loss_{self.model.name}.h5",
                    overwrite=True,
                )
                pd.DataFrame(self.model.history.history).to_pickle(
                    f"{self.model.name}/history_lowest_loss_epoch_{self.model.name}.pkl"
                )

            lr = K.get_value(self.model.optimizer.lr)
            self.learning_rates.append(lr)
            self.model.save(
                f"{self.model.name}/model_{self.model.name}_newest_epoch.h5",
                overwrite=True,
            )
            # self.model.save(f'{self.model.name}/model_{self.model.name}_newest_epoch', overwrite=True)

            self.model.save_weights(
                f"{self.model.name}/weights_{self.model.name}_newest_epoch.h5",
                overwrite=True,
            )
            pd.DataFrame(self.model.history.history).to_pickle(
                f"{self.model.name}/history_newest_epoch_{self.model.name}.pkl"
            )

        if epoch % 100 == 0:
            self.model.save(
                f"{self.model.name}/model_{self.model.name}_epoch_{epoch}.h5",
                overwrite=True,
            )



class PerformancePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test, model_name):
        self.x_test = x_test
        self.y_test = y_test
        self.model_name = model_name
        
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_test)
        fig, ax = plt.subplots(figsize=(8,4))
        plt.scatter(y_test, y_pred, alpha=0.6, 
            color='#FF0000', lw=1, ec='black')
        
        lims = [0, 5]

        plt.plot(lims, lims, lw=1, color='#0000FF')
        plt.ticklabel_format(useOffset=False, style='plain')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(lims)
        plt.ylim(lims)

        plt.tight_layout()
        plt.title(f'Prediction Visualization Keras Callback - Epoch: {epoch}')
        plt.savefig('model_train_images/'+self.model_name+"_"+str(epoch))
        plt.close()