import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Input,
    TimeDistributed,
    Conv1D,
    Cropping1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error

"""
Fixing randomness
"""
SEED = 42
import random

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class TimeDependentDeepLearningAnomalyDetection:
    def __init__(self):

        # Parameters for time dependent analysis
        self.window = 50
        self.stride = 10
        self.telescope = 3
        self.features = None
        self.model = None
        pass

    def build_train_test_sets(self, df):
        """
        This method creates the train and test sets from the given dataframe `df`.
        The dataframe is expected to have a time series data format.
        """

        test_size = int(df.shape[0]*0.2)

        # Split the data into train and test sets
        X_train_raw = df.iloc[:-test_size]  # All rows except the last `test_size` rows
        X_test_raw = df.iloc[-test_size:]   # The last `test_size` rows
        
        # Print the shapes to confirm the split
        print("Train shape:", X_train_raw.shape)
        print("Test shape:", X_test_raw.shape)
        
        # Normalize both train and test sets using min-max normalization
        X_min = X_train_raw.min()   # Minimum of each column
        X_max = X_train_raw.max()   # Maximum of each column
        
        # Normalize train and test sets using the training data min and max
        X_train_raw = (X_train_raw - X_min) / (X_max - X_min)
        X_test_raw = (X_test_raw - X_min) / (X_max - X_min)

        X_train, y_train = self.build_sequences(X_train_raw)
        X_test, y_test = self.build_sequences(X_test_raw)
        #X_train.shape, y_train.shape, X_test.shape, y_test.shape
        X_train = np.nan_to_num(X_train, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0)

        return X_train, X_test, y_train, y_test


    def build_sequences(self, df):
        """
        Creates sequences of data (features) and future data (as labels) from the given DataFrame.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame containing the data.
        window (int): The size of the sliding window to capture sequences of data.
        stride (int): The number of rows to skip between windows.
        telescope (int): The number of rows into the future to use as the next sequence.
        
        Returns:
        tuple: Two numpy arrays containing the sequences (features) and corresponding future data (labels).
        """
        # Sanity check to avoid runtime errors
        window = self.window
        stride = self.stride
        telescope = self.telescope
        assert window % stride == 0, "Window size must be divisible by the stride."
        
        dataset = []
        labels = []
        
        # Convert dataframe to numpy array
        temp_df = df.copy().values
    
        # Calculate if padding is needed
        padding_check = len(df) % window
        if padding_check != 0:
            # Compute padding length
            padding_len = window - padding_check
            # Pad the features with zeros to make the length divisible by the window size
            padding = np.zeros((padding_len, temp_df.shape[1]), dtype='float32')
            temp_df = np.concatenate((padding, temp_df))
            assert len(temp_df) % window == 0
    
        # Iterate through the dataframe with the given stride
        for idx in np.arange(0, len(temp_df) - window - telescope, stride):
            # Extract the window of features
            dataset.append(temp_df[idx:idx + window])
            # Extract the label window after the current window
            labels.append(temp_df[idx + window:idx + window + telescope])
    
        # Convert to numpy arrays for easier handling
        dataset = np.array(dataset)
        labels = np.array(labels)
        
        return dataset, labels


    def build_model(self, input_shape, output_shape):
        """
        Builds an MLP model using TensorFlow/Keras that takes a 3D input and outputs a 3D output.
        
        Parameters:
        input_shape (tuple): Shape of the input data (e.g., (200, 9)).
        output_shape (tuple): Shape of the output data (e.g., (10, 9)).
        
        Returns:
        model (tf.keras.Model): Compiled MLP model.
        """
        model = models.Sequential()
        
        # Input Layer: Flatten input shape (200, 9) into (200*9,)
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(layers.Flatten())  # Flatten the 3D input into a 1D vector (200 * 9 = 1800)
    
        # Hidden Layers (add more layers as necessary)
        model.add(layers.Dense(256, activation='relu'))  # 1st hidden layer with 256 neurons
        model.add(layers.Dense(128, activation='relu'))  # 2nd hidden layer with 128 neurons
        
        # Output Layer: Dense layer with output size matching output_shape (10*9 = 90)
        model.add(layers.Dense(output_shape[0] * output_shape[1], activation='linear'))  # Regression case
        
        # Reshape back to the original output shape (10, 9)
        model.add(layers.Reshape(output_shape))
    
        # Compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # MSE for regression
        
        return model

    def build_lstm_model(self, input_shape, output_shape):
        """
        Builds an LSTM model using TensorFlow/Keras that takes a 3D input and outputs a 3D output.
        
        Parameters:
        input_shape (tuple): Shape of the input data (e.g., (200, 9)).
        output_shape (tuple): Shape of the output data (e.g., (10, 9)).
        
        Returns:
        model (tf.keras.Model): Compiled LSTM model.
        """
        model = models.Sequential()
        
        # Add LSTM layers
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(layers.LSTM(128, return_sequences=True))  # LSTM layer with 128 units
        model.add(layers.LSTM(64, return_sequences=True))   # LSTM layer with 64 units
        
        # Add Dense layer to produce the final output shape
        model.add(layers.TimeDistributed(layers.Dense(output_shape[1], activation='linear')))  # Dense layer with output_dim matching second dimension of output_shape
    
        model.add(layers.Lambda(lambda x: x[:, -output_shape[0]:, :]))
        # Compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # MSE for regression
        
        return model


    def build_transformer_model(self, input_shape, output_shape, num_heads=4, ff_dim=128, num_blocks=2):
        """
        Builds a Transformer model using TensorFlow/Keras.
        
        Parameters:
        input_shape (tuple): Shape of the input data (e.g., (200, 9)).
        output_shape (tuple): Shape of the output data (e.g., (10, 9)).
        num_heads (int): Number of attention heads in MultiHeadAttention.
        ff_dim (int): Dimension of the feed-forward network.
        num_blocks (int): Number of Transformer blocks.
        
        Returns:
        model (tf.keras.Model): Compiled Transformer model.
        """
        inputs = layers.Input(shape=input_shape)
        
        # Positional Encoding
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        positions = tf.expand_dims(positions, 0)  # Shape: (1, seq_len)
        position_embeddings = layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[1])(positions)
        
        x = inputs + position_embeddings
        
        for _ in range(num_blocks):
            # Multi-Head Self-Attention
            attn_output = layers.MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=input_shape[1]
            )(x, x)
            attn_output = layers.Dropout(0.1)(attn_output)
            x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # Feed Forward Network
            ff_output = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
            ff_output = layers.Conv1D(filters=input_shape[1], kernel_size=1)(ff_output)
            ff_output = layers.Dropout(0.1)(ff_output)
            x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
        
        # Dense Layer to produce the final output shape
        x = layers.TimeDistributed(layers.Dense(output_shape[1], activation='linear'))(x)
    
        x = layers.Lambda(lambda x: x[:, -output_shape[0]:, :])(x)
        
        # Compile the model
        model = models.Model(inputs=inputs, outputs=x)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model



    def fit(self, X_train, y_train, epochs=100, batch_size=32):
        
        #self.model = self.build_model(input_shape, output_shape)
        #mlp.summary()
        # Define early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=50,         # Stop training if no improvement after 10 epochs
            restore_best_weights=True  # Restore the model weights from the epoch with the best loss
        )
        
        # Train the model and store the history
        history = self.model.fit(
            X_train, y_train,            # Training data
            validation_split=0.2,        # Use 20% of the data for validation
            epochs=2000,                  # Maximum number of epochs to train
            batch_size=32,               # Batch size
            callbacks=[early_stopping],  # Early stopping callback
            verbose=1                    # Show progress
        )

        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # If you have other metrics, like MAE (mean absolute error), you can plot them too
        if 'mae' in history.history:
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['mae'], label='Training MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title('Training and Validation MAE Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True)
            plt.show()


    def predict(self, X_test, y_test):
        """
        This method makes predictions using the trained model.
        """
        test_loss, test_mae = self.model.evaluate(X_test, y_test)
        
        # Make predictions on the test set
        y_pred = self.model.predict(X_test)
        
        return y_pred, test_loss, test_mae
        

    def detect_anomalies(self, df, verbose=0):
        """
        This method analyzes the data using the trained model.

        Args:
            df (pd.DataFrame): The dataframe containing the time series data.

        Returns:
            float: The mean squared error (MSE) of the predictions.
        """
        X_train, X_test, y_train, y_test = self.build_train_test_sets(df)

        # Get input and output shapes for model building
        input_shape = X_train.shape[1:]  # Exclude batch size dimension
        output_shape = y_train.shape[1:]

        self.model = self.build_model(input_shape, output_shape)
        self.model.summary()
        self.fit(X_train, y_train)
        y_pred, test_loss, test_mae = self.predict(X_test, y_test)

        if verbose:
            print(f"Test Loss: {test_loss}")
            print(f"Test MAE: {test_mae}")
        return test_loss, test_mae


    def detect_anomalies_LSTM(self, df, verbose=0):
        """
        """
        X_train, X_test, y_train, y_test = self.build_train_test_sets(df)

        # Get input and output shapes for model building
        input_shape = X_train.shape[1:]  # Exclude batch size dimension
        output_shape = y_train.shape[1:]

        self.model = self.build_lstm_model(input_shape, output_shape)
        self.model.summary()
        self.fit(X_train, y_train)
        y_pred, test_loss, test_mae = self.predict(X_test, y_test)

        if verbose:
            print(f"Test Loss: {test_loss}")
            print(f"Test MAE: {test_mae}")
        return test_loss, test_mae


    def detect_anomalies_transformer(self, df, verbose=0):
        """
        """
        X_train, X_test, y_train, y_test = self.build_train_test_sets(df)

        # Get input and output shapes for model building
        input_shape = X_train.shape[1:]  # Exclude batch size dimension
        output_shape = y_train.shape[1:]

        self.model = self.build_transformer_model(input_shape, output_shape)
        self.model.summary()
        self.fit(X_train, y_train)
        y_pred, test_loss, test_mae = self.predict(X_test, y_test)

        if verbose:
            print(f"Test Loss: {test_loss}")
            print(f"Test MAE: {test_mae}")
        return test_loss, test_mae
