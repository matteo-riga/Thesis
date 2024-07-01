import os
import warnings
import numpy as np
import logging
import random

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)
print(tf.__version__)

class DeepLearningAnomalyDetection():
    def __init__(self, X, verbose=0):
        self.X = X
        self.verbose = verbose
        
        # fix randomness
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=Warning)
        np.random.seed(seed)
        random.seed(seed)
        
    def split(self):
        X = self.X
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, random_state=seed, test_size=.25, stratify=np.argmax(y,axis=1))
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, random_state=seed, test_size=len(X_test), stratify=np.argmax(y_train_val,axis=1))
        
        if self.verbose:
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        self.input_shape = X_train.shape[1:]
        self.output_shape_not_expanded = y_train.shape[1:]
        self.output_shape = np.expand_dims(output_shape_not_expanded, axis=-1)
        self.batch_size = 32
        self.epochs = 1000
        self.learning_rate = 1e-2

        if self.verbose:
            print(f"Input Shape: {input_shape}, Output Shape: {output_shape}, Batch Size: {batch_size}, Epochs: {epochs}")

        return X_train, y_train, X_val, y_val, X_test, y_test


    def build_autoencoder(input_shape=self.input_shape, output_shape=self.output_shape, seed=seed):
        
        input_layer = tfkl.Input(shape=input_shape, name='Input')

        conv1 = tfkl.Conv2D(
            filters=6,
            kernel_size=(5,5),
            padding='same',
            activation='tanh',
            name='conv1'
        )(input_layer)
    
        pool1 = tfkl.MaxPooling2D(
            pool_size=(2,2),
            name='mp1'
        )(conv1)
    
        conv2 = tfkl.Conv2D(
            filters=16,
            kernel_size=(5,5),
            padding='valid',
            activation='tanh',
            name='conv2'
        )(pool1)
    
        pool2 = tfkl.MaxPooling2D(
            pool_size =(2,2),
            name='mp2'
        )(conv2)
            flattening_layer=tfkl.Flatten(
        name='flatten'
        )(pool2)
    
        classifier_layer=tfkl.Dense(
            units=120,
            activation='tanh',
            name='dense1'
        )(flattening_layer)
    
        classifier_layer = tfkl.Dense(
            units=84,
            activation='tanh',
            name='dense2'
        )(classifier_layer)
    
        output_layer = tfkl.Dense(
            units=output_shape,
            activation='softmax',
            name='Output'
        )(classifier_layer)

        # Connect input and output through the Model class
        model = tfk.Model(inputs=input_layer, outputs=output_layer, name='LeNet')

        # Compile the model
        model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics=['accuracy'])
    
        # Return the model
        return model
    
        
    def initialize_model_and_plot(self):
        self.model_fct = build_autoencoder
        self.model = self.model_fct(self.input_shape, self.output_shape)
        self.model.summary()
        if self.verbose:
            tfk.utils.plot_model(self.model, expand_nested=True, show_shapes=True)

    def train(self):
        self.history = self.model.fit(
            x = self.X_train,
            y = self.y_train,
            batch_size = self.batch_size,
            epochs = self.epochs,
            validation_data = (self.X_val, self.y_val)
        ).history

        if self.verbose:
            plt.figure(figsize=(15,5))
            plt.plot(self.history['loss'], alpha=.3, color='#ff7f0e', linestyle='--')
            plt.plot(self.history['val_loss'], label='LeNet', alpha=.8, color='#ff7f0e')
            plt.legend(loc='upper left')
            plt.title('Categorical Crossentropy')
            plt.grid(alpha=.3)
            
            plt.figure(figsize=(15,5))
            plt.plot(self.history['accuracy'], alpha=.3, color='#ff7f0e', linestyle='--')
            plt.plot(self.history['val_accuracy'], label='LeNet', alpha=.8, color='#ff7f0e')
            plt.legend(loc='upper left')
            plt.title('Accuracy')
            plt.grid(alpha=.3)
            
            plt.show()