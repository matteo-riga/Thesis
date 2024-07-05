import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns


'''
Fixing randomness
'''
SEED = 42
import random
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class DeepLearningAnomalyDetection():

    def __init__(self):
        pass


    def build_autoencoder(self, input_dim, encoding_dim):
        '''
        Function that builds and returns an autoencoder model

        Args:
            input_dim (int): iput shape
            encoding_dim (int): dimension of latent space

        Returns:
            autoencoder (Model): autoencoder model
        '''
        
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        encoder_layer1 = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
        decoder_layer1 = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoder_layer1)
        autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder_layer1)
        
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
        return autoencoder


    def build_vae(self, input_shape, encoding_dim):
        '''
        Function that builds and returns a VAE autoencoder model

        Args:
            input_shape (int): iput shape
            encoding_dim (int): dimension of latent space

        Returns:
            vae (Model): VAE autoencoder model
        '''
        # Encoder
        inputs = Input(shape=(input_shape,))
        h = Dense(encoding_dim, activation='relu')(inputs)
        
        # Latent space
        z_mean = Dense(encoding_dim)(h)
        z_log_var = Dense(encoding_dim)(h)
        
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = Lambda(sampling, output_shape=(encoding_dim,))([z_mean, z_log_var])
        
        # Decoder
        decoder_h = Dense(encoding_dim, activation='relu')
        decoder_mean = Dense(input_shape, activation='sigmoid')
        h_decoded = decoder_h(z)
        outputs = decoder_mean(h_decoded)
        
        # VAE model
        vae = Model(inputs, outputs)
        
        # Add VAE loss
        outputs = VAELossLayer(input_shape)([inputs, outputs, z_mean, z_log_var])
        vae = Model(inputs, outputs)
        
        vae.compile(optimizer='adam')
        
        return vae



    def get_dataset(self, dataframe):
        dataset = dataframe[['severity_scores', 'timedelta', 'log key']]
        return dataset


    def train_test_model(self, normal_df, anomalous_df, model, plots=[0,0,0,0]):

        normal_dataset = self.get_dataset(normal_df)
        anomalous_dataset = self.get_dataset(anomalous_df)
        # Normalize the data
        scaler = MinMaxScaler()
        normal_data = scaler.fit_transform(normal_dataset)
        anomalous_data = scaler.transform(anomalous_dataset)
        
        # Define the autoencoder model
        input_dim = normal_data.shape[1]
        encoding_dim = 3  # You can adjust the encoding dimension as needed

        if model=='autoencoder':
            model = self.build_autoencoder(input_dim, encoding_dim)
        elif model=='vae':
            model = self.build_vae(input_dim, encoding_dim)
            
        #model.summary()
        
        # Define the EarlyStopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train the autoencoder with the EarlyStopping callback
        history = model.fit(normal_data, normal_data,
                                  epochs=50,
                                  batch_size=32,
                                  shuffle=True,
                                  validation_split=0.2,
                                  callbacks=[early_stopping],
                                  verbose=0
                            )


        if plots[0]:
            # Plot the training and validation loss
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()


        
        # Use the model to reconstruct the normal and anomalous data
        normal_reconstruction = model.predict(normal_data, verbose=0)
        anomalous_reconstruction = model.predict(anomalous_data, verbose=0)
        
        # Calculate the reconstruction error
        normal_reconstruction_error = np.mean(np.square(normal_data - normal_reconstruction), axis=1)
        anomalous_reconstruction_error = np.mean(np.square(anomalous_data - anomalous_reconstruction), axis=1)

        # Determine a threshold for anomaly detection (e.g., based on the 95th percentile of the reconstruction error)
        threshold = np.percentile(normal_reconstruction_error, 99.75)
        
        # Identify anomalies
        anomalies = anomalous_reconstruction_error > threshold

        # Identify normal samples misclassified as anomalies
        false_positive = normal_reconstruction_error > threshold
        
        # Create arrays to represent true anomalies and reconstructed anomalies
        true_normal_label = np.zeros(normal_reconstruction_error.shape[0])
        true_anomaly_label = np.ones(anomalous_reconstruction_error.shape[0])
        true_anomalies = np.concatenate((true_normal_label, true_anomaly_label))
        reconstructed_anomalies = np.concatenate((false_positive, anomalies))


        if plots[1]:
            # Plot the reconstruction error
            print(f"Mean normal reconstruction error: {np.mean(normal_reconstruction_error)}")
            print(f"Mean anomalous reconstruction error: {np.mean(anomalous_reconstruction_error)}")
            
            plt.figure(figsize=(10, 6))
            plt.hist(normal_reconstruction_error, bins=50, alpha=0.6, label='Normal Data')
            plt.hist(anomalous_reconstruction_error, bins=50, alpha=0.6, label='Anomalous Data')
            plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Frequency')
            plt.legend()
            plt.title('Reconstruction Error Histogram')
            plt.show()
            
            # Print the results
            print("Threshold for anomaly detection:", threshold)
            print("Number of anomalies detected:", np.sum(anomalies))
            print("Number of false positives detected:", np.sum(false_positive))


        if plots[2]:
            print(f"Mean normal reconstruction error: {np.mean(normal_reconstruction_error)}")
            print(f"Mean anomalous reconstruction error: {np.mean(anomalous_reconstruction_error)}")
            # Create bins for log-log histogram
            bins = np.logspace(np.log10(min(normal_reconstruction_error.min(), anomalous_reconstruction_error.min())),
                               np.log10(max(normal_reconstruction_error.max(), anomalous_reconstruction_error.max())), 50)
            
            # Plot the reconstruction error histogram with log-log scale
            plt.figure(figsize=(10, 6))
            plt.hist(normal_reconstruction_error, bins=bins, alpha=0.6, label='Normal Data')
            plt.hist(anomalous_reconstruction_error, bins=bins, alpha=0.6, label='Anomalous Data')
            plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Frequency')
            plt.legend()
            plt.title('Reconstruction Error Histogram (Log-Log Scale)')
            plt.show()
            
            # Print the results
            print("Threshold for anomaly detection:", threshold)
            print("Number of anomalies detected:", np.sum(anomalies))
            print("Number of false positives detected:", np.sum(false_positive))

        
        if plots[3]:
            # Example ground truth and predictions (replace with your actual data)
            ground_truth = true_anomalies
            predictions = reconstructed_anomalies
            
            # Calculate confusion matrix
            cm = confusion_matrix(ground_truth, predictions)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Normal', 'Anomaly'], 
                        yticklabels=['Normal', 'Anomaly'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.show()


        return true_anomalies, reconstructed_anomalies, false_positive


    def ensemble_method(self, normal_df, anomalous_df, plots=[0]):
        
        true_anomalies, pred_autoencoder = self.train_test_model(normal_df, anomalous_df, 'autoencoder', plots=[0,0,0,0])
        true_anomalies, pred_vae = self.train_test_model(normal_df, anomalous_df, 'vae', plots=[0,0,0,0])
        combined_predictions = np.logical_and(pred_autoencoder, pred_vae).astype(int)
        threshold = 0.5  # Simple majority voting
        # Convert to final anomaly predictions based on threshold
        final_predictions = (combined_predictions >= threshold).astype(int)
        # Example ground truth and predictions (replace with your actual data)
        ground_truth = true_anomalies
        predictions = final_predictions
                    
        # Calculate confusion matrix
        cm = confusion_matrix(ground_truth, predictions)

        if plots[0]:
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                    xticklabels=['Normal', 'Anomaly'], 
                                    yticklabels=['Normal', 'Anomaly'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.show()



'''
Additional classes needed
'''

class VAELossLayer(Layer):
    def __init__(self, input_dim, **kwargs):
        self.input_dim = input_dim
        super(VAELossLayer, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        x, x_decoded_mean, z_mean, z_log_var = inputs
        reconstruction_loss = tf.reduce_mean(tf.square(x - x_decoded_mean)) * self.input_dim
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(vae_loss)
        return x_decoded_mean