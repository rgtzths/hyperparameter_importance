import keras_tuner
import keras
import tensorflow as tf
import numpy as np
from sklearn.metrics import matthews_corrcoef, mean_squared_error, accuracy_score, f1_score, mean_absolute_error, r2_score
from skimage.color import rgb2gray
import time

from tensorflow.keras.utils import Sequence
import numpy as np   

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

def get_optimizer(optimizer):

    if optimizer.lower() == "adam":
        return keras.optimizers.Adam
    elif optimizer.lower() == "rmsprop":
        return keras.optimizers.RMSprop
    elif optimizer.lower() == "sgd":
        return keras.optimizers.SGD
    elif optimizer.lower() == "adadelta":
        return keras.optimizers.Adadelta
    elif optimizer.lower() == "adagrad":
        return keras.optimizers.Adagrad
    elif optimizer.lower() == "adamax":
        return keras.optimizers.Adamax
    elif optimizer.lower() == "nadam":
        return keras.optimizers.Nadam
    elif optimizer.lower() == "ftrl":
        return keras.optimizers.Ftrl
    else:
        raise ValueError(f'The optimizer {optimizer} is not supported!')

class DenseModel(keras_tuner.HyperModel):

    def __init__(self, hyperparameters):
        super().__init__()
        self.is_categorical = False
        self.hyperparameters = hyperparameters
        if "binary_crossentropy" in hyperparameters["loss"] or \
            "sparse_categorical_crossentropy" in hyperparameters["loss"] or \
            "categorical_crossentropy" in hyperparameters["loss"]:
            self.is_categorical = True
        
        self.metrics = ['accuracy'] if self.is_categorical else ['mean_squared_error']# Add new_metrics


    def build(self, hp):
        inputs = keras.Input(shape=(self.hyperparameters["input_shape"]))
        outputs = None

        optimizer = hp.Choice("optimizer", self.hyperparameters["optimizer"])
        learning_rate = hp.Float("learning_rate",  min_value=self.hyperparameters["lr"][0],
                                  max_value=self.hyperparameters["lr"][1], 
                                  step=self.hyperparameters["lr"][2])
        
        self.loss = hp.Choice("loss", self.hyperparameters["loss"])
        
        n_nodes = hp.Int(f"n_dense_nodes",  min_value=self.hyperparameters["n_dense_nodes"][0], 
                            max_value=self.hyperparameters["n_dense_nodes"][1], 
                            step=self.hyperparameters["n_dense_nodes"][2])
        
        activation = hp.Choice(f"dense_activation", self.hyperparameters["dense_activation_fn"])

        dropout_rate = hp.Float(f"dense_dropout_rate", min_value=self.hyperparameters["dense_dropout"][0], 
                                max_value=self.hyperparameters["dense_dropout"][1], 
                                step=self.hyperparameters["dense_dropout"][2])
            
        for i in range(hp.Int("n_dense_layers", min_value=self.hyperparameters["n_dense_layers"][0], 
                              max_value=self.hyperparameters["n_dense_layers"][1], 
                              step=self.hyperparameters["n_dense_layers"][2])):
            if outputs == None:
                outputs = keras.layers.Dense(n_nodes, activation=activation)(inputs)
            else:
                outputs = keras.layers.Dense(n_nodes, activation=activation)(outputs)

            outputs = keras.layers.Dropout(dropout_rate)(outputs)
        

        if self.loss == "binary_crossentropy":
            outputs = keras.layers.Dense(units=1, activation="sigmoid")(outputs)
        elif self.loss in "sparse_categorical_crossentropy":
            outputs = keras.layers.Dense(units=self.hyperparameters["n_outputs"], activation="sigmoid")(outputs)
        else:
            outputs = keras.layers.Dense(units=self.hyperparameters["n_outputs"], activation="linear")(outputs)
            
        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=get_optimizer(optimizer)(learning_rate=learning_rate),
            loss=self.loss,
            metrics=self.metrics, 
        )
        return model

    def preprocess_data(self, x, y, validation_data, batch_size):
        ##Add transformations based on the dataset we use
        ##Flatten images, remove time dependencies etc.

        if self.loss == "categorical_crossentropy":
            y = keras.utils.to_categorical(y)
            if validation_data:
                x_val, y_val = validation_data
                y_val = keras.utils.to_categorical(y_val)
                validation_data = (x_val, y_val)

        if self.hyperparameters["dataset"] in ["cifar10", "cifar100", "mnist", "fashion_mnist", "license_plate", "utk_faces", "mri"]:
            x_new = np.zeros((x.shape[0], x.shape[1]*x.shape[2]))

            
            for i in range(len(x)):
                x_new[i, :] = rgb2gray(x[i,:]).flatten() if self.hyperparameters["dataset"] in ["cifar10", "cifar100","license_plate", "utk_faces", "mri"] else x[i,:].flatten()
            
            x = x_new

            if validation_data:
                x_val, y_val = validation_data
                x_val_new = np.zeros((x_val.shape[0], x_val.shape[1]*x_val.shape[2]))

                for i in range(len(x_val)):
                    x_val_new[i, :] = rgb2gray(x_val[i,:]).flatten() if self.hyperparameters["dataset"] in ["cifar10", "cifar100", "license_plate", "utk_faces", "mri"] else x_val[i,:].flatten()

                validation_data = (x_val_new, y_val)

        return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size), tf.data.Dataset.from_tensor_slices(validation_data).batch(batch_size)

    #In case we want to optimize anything of the training
    def fit(self, hp, model, x, y, validation_data=None, *args, **kwargs):

        batch_size = hp.Int("batch_size", min_value=self.hyperparameters["batch_size"][0], 
                                        max_value=self.hyperparameters["batch_size"][1], 
                                        step=self.hyperparameters["batch_size"][2])
        
        train_data, val_data = self.preprocess_data(x, y, validation_data, batch_size)

        #kwargs["callbacks"].append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=30))
        
        train_time = time.time()

        model.fit(
            x=train_data,
            batch_size= batch_size,
            epochs = self.hyperparameters["epochs"],
            validation_data=val_data,
            verbose = 0,
            shuffle=True,
            **kwargs
        )
        train_time = time.time() -  train_time

        if self.is_categorical:
            infer_time = time.time()
            predictions = model.predict(val_data, verbose=0)
            infer_time = time.time() - infer_time

            predictions = [round(x[0]) for x in predictions] if self.loss == "binary_crossentropy" \
                            else [np.argmax(x) for x in predictions]
            
            results = {'mcc' : matthews_corrcoef(validation_data[1], predictions),
                    'acc' : accuracy_score(validation_data[1], predictions), 
                    'f1-score' : f1_score(validation_data[1], predictions, average="macro"),
                    'train_time' : train_time,
                    'infer_time' : infer_time}
            
            del train_data
            del val_data
            del validation_data
            del predictions

            return results
        else:
            infer_time = time.time()
            predictions = model.predict(val_data, verbose=0)
            infer_time = time.time() - infer_time
            
            if predictions.shape[1] < 2:
                predictions = predictions.reshape((-1,))
            results = {'mse' :  mean_squared_error(validation_data[1], predictions),
                    'mae' : mean_absolute_error(validation_data[1], predictions),
                    'train_time' : train_time,
                    'infer_time' : infer_time}

            del train_data
            del val_data
            del validation_data
            del predictions
            
            return results
        

class CNNModel(keras_tuner.HyperModel):

    def __init__(self, hyperparameters):
        super().__init__()

        self.is_categorical = False
        self.hyperparameters = hyperparameters
        if "binary_crossentropy" in hyperparameters["loss"] or \
            "sparse_categorical_crossentropy" in hyperparameters["loss"] or \
            "categorical_crossentropy" in hyperparameters["loss"]:
            self.is_categorical = True
        
        self.metrics = ['accuracy'] if self.is_categorical else ['mean_squared_error']# Add new_metrics

    def build(self, hp):

        if self.hyperparameters["dimension"] == 1:
            conv_layer = keras.layers.Conv1D
            maxpooling_layer = keras.layers.MaxPooling1D

        elif self.hyperparameters["dimension"] == 2:
            conv_layer = keras.layers.Conv2D
            maxpooling_layer = keras.layers.MaxPooling2D

        else:
            conv_layer = keras.layers.Conv3D
            maxpooling_layer = keras.layers.MaxPooling3D

        inputs = keras.Input(shape=(self.hyperparameters["input_shape"]))
        outputs = None

        optimizer = hp.Choice("optimizer", self.hyperparameters["optimizer"])
        learning_rate = hp.Float("learning_rate",  min_value=self.hyperparameters["lr"][0],
                                  max_value=self.hyperparameters["lr"][1], 
                                  step=self.hyperparameters["lr"][2])
        
        self.loss = hp.Choice("loss", self.hyperparameters["loss"])
        
        n_kernels = hp.Int(f"n_kernels",  min_value=self.hyperparameters["n_kernels"][0], 
                            max_value=self.hyperparameters["n_kernels"][1], 
                            step=self.hyperparameters["n_kernels"][2])
        
        kernel_size = hp.Int(f"kernel_size",  min_value=self.hyperparameters["kernel_size"][0], 
                            max_value=self.hyperparameters["kernel_size"][1], 
                            step=self.hyperparameters["kernel_size"][2])
        
        pool_size = hp.Int(f"pool_size",  min_value=self.hyperparameters["pool_size"][0], 
                            max_value=self.hyperparameters["pool_size"][1], 
                            step=self.hyperparameters["pool_size"][2])
        
        activation = hp.Choice(f"conv_activation", self.hyperparameters["conv_activation_fn"])

        dropout_rate = hp.Float(f"conv_dropout_rate", min_value=self.hyperparameters["conv_dropout"][0], 
                                max_value=self.hyperparameters["conv_dropout"][1], 
                                step=self.hyperparameters["conv_dropout"][2])
        
        for i in range(hp.Int("n_conv_layers", min_value=self.hyperparameters["n_conv_layers"][0], 
                              max_value=self.hyperparameters["n_conv_layers"][1], 
                              step=self.hyperparameters["n_conv_layers"][2])):
            
            if outputs == None:
                outputs = conv_layer(n_kernels, 
                    kernel_size=kernel_size, 
                    activation=activation)(inputs)
            else:
                outputs = conv_layer(n_kernels, 
                    kernel_size=kernel_size, 
                    activation=activation)(outputs)
                
        outputs = maxpooling_layer(pool_size=pool_size)(outputs)
        outputs = keras.layers.BatchNormalization()(outputs)
        outputs = keras.layers.Flatten()(outputs)
        outputs = keras.layers.Dropout(dropout_rate)(outputs)

        
        n_nodes = hp.Int(f"n_dense_nodes",  min_value=self.hyperparameters["n_dense_nodes"][0], 
                            max_value=self.hyperparameters["n_dense_nodes"][1], 
                            step=self.hyperparameters["n_dense_nodes"][2])
        
        activation = hp.Choice(f"dense_activation", self.hyperparameters["dense_activation_fn"])

        dropout_rate = hp.Float(f"dense_dropout_rate", min_value=self.hyperparameters["dense_dropout"][0], 
                                max_value=self.hyperparameters["dense_dropout"][1], 
                                step=self.hyperparameters["dense_dropout"][2])
            
        for i in range(hp.Int("n_dense_layers", min_value=self.hyperparameters["n_dense_layers"][0], 
                              max_value=self.hyperparameters["n_dense_layers"][1], 
                              step=self.hyperparameters["n_dense_layers"][2])):
            
            outputs = keras.layers.Dense(n_nodes, activation=activation)(outputs)

            outputs = keras.layers.Dropout(dropout_rate)(outputs)
        
        if self.loss == "binary_crossentropy":
            outputs = keras.layers.Dense(units=1, activation="sigmoid")(outputs)
        elif self.loss in "sparse_categorical_crossentropy":
            outputs = keras.layers.Dense(units=self.hyperparameters["n_outputs"], activation="sigmoid")(outputs)
        else:
            outputs = keras.layers.Dense(units=self.hyperparameters["n_outputs"], activation="linear")(outputs)

        model = keras.Model(inputs, outputs)
        
        model.compile(
            optimizer=get_optimizer(optimizer)(learning_rate=learning_rate),
            loss=self.loss,
            metrics=self.metrics, 
        )
        return model

    def preprocess_data(self, x, y, validation_data, batch_size):
        ##Add transformations based on the dataset we use
        ##Flatten images, remove time dependencies etc.

        if self.loss == "categorical_crossentropy":
            y = keras.utils.to_categorical(y)
            if validation_data:
                x_val, y_val = validation_data
                y_val = keras.utils.to_categorical(y_val)
                validation_data = (x_val, y_val)
         
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size), tf.data.Dataset.from_tensor_slices(validation_data).batch(batch_size)

    
    #In case we want to optimize anything of the training
    def fit(self, hp, model, x, y, validation_data=None, **kwargs):

        batch_size = hp.Int("batch_size", min_value=self.hyperparameters["batch_size"][0], 
                                        max_value=self.hyperparameters["batch_size"][1], 
                                        step=self.hyperparameters["batch_size"][2])

        train_data, val_data = self.preprocess_data(x, y, validation_data, batch_size)

        #kwargs["callbacks"].append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=30))

        train_time = time.time()
        model.fit(
            train_data,
            batch_size=batch_size,
            epochs = self.hyperparameters["epochs"],
            validation_data=val_data,
            verbose = 0,
            shuffle=True,
            **kwargs
        )
        train_time = time.time() - train_time


        if self.is_categorical:
            infer_time = time.time()
            predictions = model.predict(val_data, verbose=0)
            infer_time = time.time() - infer_time

            predictions = [round(x[0]) for x in predictions] if self.loss == "binary_crossentropy" \
                            else [np.argmax(x) for x in predictions] 

            results = {'mcc' : matthews_corrcoef(validation_data[1], predictions), 
                    'acc' : accuracy_score(validation_data[1], predictions), 
                    'f1-score' : f1_score(validation_data[1], predictions, average="macro"),
                    'train_time' : train_time,
                    'infer_time' : infer_time}

            del train_data
            del val_data
            del validation_data
            del predictions

            return results
        else:
            infer_time = time.time()
            predictions = model.predict(val_data, verbose=0)
            infer_time = time.time() - infer_time

            if predictions.shape[1] < 2:
                predictions = predictions.reshape((-1,))
                
            results = {'mse' : mean_squared_error(validation_data[1], predictions),
                    'mae' : mean_absolute_error(validation_data[1], predictions),
                    'train_time' : train_time,
                    'infer_time' : infer_time}
            
            del train_data
            del val_data
            del validation_data
            del predictions

            return results

