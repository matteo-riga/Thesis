import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
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
        self.telescope = 10
        self.features = None
        self.model = None
        pass

    def build_train_test_sets(self, df):
        """
        This method creates the train and test sets from the given dataframe `df`.
        The dataframe is expected to have a time series data format.
        """

        self.features = len(df.columns)

        X, y = self.create_sequences(df)

        # Calculate the split index based on a percentage
        split_index = int(len(X) * 0.8)

        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        return X_train, X_test, y_train, y_test

    def create_sequences(self, df):
        """
        This method creates sequences of data for the given dataframe `df`.
        The sequences are used to feed the deep learning model.
        """
        X = []
        y = []

        for i in range(len(df) - self.window - self.telescope + 1):
            X.append(df.iloc[i : i + self.window].values)
            y.append(df.iloc[i + self.window : i + self.window + self.telescope].values)

        X = np.array(X)
        y = np.array(y)

        return X, y

    def build_model(self, input_shape, output_shape):
        """
        This method builds a simple one-layer LSTM model.
        """
        input_layer = Input(shape=(self.window, self.features))
        x = LSTM(16, activation="relu", return_sequences=True)(input_layer)
        # output_layer = TimeDistributed(Dense(self.features))(x)
        output_layer = Conv1D(
            self.telescope, self.features, padding="same", name="output_layer"
        )(x)
        crop_size = self.window - self.telescope
        # Crop the output to the desired length
        output_layer = Cropping1D((0, crop_size), name="cropping")(output_layer)

        model = Model(inputs=input_layer, outputs=output_layer, name="CONV_LSTM_model")

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        self.model = model
        return model

    def fit(self, X_train, y_train, epochs=10, batch_size=32):
        """
        This method trains the deep learning model on the training data.
        """
        # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))  # Reshape for LSTM input
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X_test):
        """
        This method makes predictions using the trained model.
        """
        # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Reshape for LSTM input
        y_pred = self.model.predict(X_test)
        return y_pred

    def evaluate(self, y_test, y_pred):
        """
        This method evaluates the model's performance.
        """
        y_test_flattened = y_test.flatten()
        y_pred_flattened = y_pred.flatten()
        mse = mean_squared_error(y_test_flattened, y_pred_flattened)
        print(f"Mean Squared Error: {mse}")
        return mse

    def analyze(self, df):
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

        m = self.build_model(input_shape, output_shape)
        m.summary()
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        mse = self.evaluate(y_test, y_pred)
        print(mse)
        return mse
