import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA, NMF, KernelPCA
from sklearn.manifold import Isomap, TSNE, LocallyLinearEmbedding, MDS

import tensorflow as tf

import umap

class ReduceDim():
    def __init__(self, n_components, df, plots = [0,0]):
        self.n_components = n_components
        self.df = df
        self.plots = plots

    def visualize(self, data):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=50)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        plt.title('Scatter Plot along the first 3 dimensions')
        plt.show()

    '''
    Function to perform PCA
    '''
    def pca(self):
        pca = PCA()
        pca.fit(self.df)
        transformed_data = pca.transform(self.df)

        if self.plots[0]:
            self.visualize(transformed_data)

        if self.plots[1]:
            plt.subplot(1, 2, 2)
            plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='-')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('Cumulative Explained Variance')
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        return transformed_data 

    '''
    Function to perform Kernel PCA
    '''
    def kern_pca(self):
        data = self.df
        kpca = KernelPCA(n_components=self.n_components, kernel='rbf')  # 'linear', 'poly', 'rbf', 'sigmoid', or 'cosine'
        X_kpca = kpca.fit_transform(data)

        if self.plots[0]:
            self.visualize(X_kpca)
            
        return X_kpca
    
    '''
    Function to perform Isomap
    '''
    def isomap(self):
        data = self.df
        n_neighbors = 5  # You can adjust the number of neighbors
        isomap = Isomap(n_components=self.n_components, n_neighbors=n_neighbors)
        X_isomap = isomap.fit_transform(data)

        if self.plots[0]:
            self.visualize(X_isomap)
            
        return X_isomap


    '''
    Function to perform LLE
    '''
    def lle(self):
        data = self.df
        n_neighbors = 10  # You can adjust the number of neighbors
        lle = LocallyLinearEmbedding(n_components=self.n_components, n_neighbors=n_neighbors, method='standard')
        X_lle = lle.fit_transform(data)

        if self.plots[0]:
            self.visualize(X_lle)
            
        return X_lle


    '''
    Function to perform MVU
    '''
    def mvu(self):
        data = self.df
        distances = pairwise_distances(data, metric='euclidean')
        mvu = MDS(n_components=self.n_components, dissimilarity='precomputed', random_state=0)
        X_mvu = mvu.fit_transform(distances)

        if self.plots[0]:
            self.visualize(X_mvu)
            
        return X_mvu


    '''
    Function to perform t-SNE
    '''
    def tsne(self):
        tsne = TSNE(n_components=self.n_components)
        tsne_components = tsne.fit_transform(self.df)

        if self.plots[0]:
            self.visualize(tsne_components)
        
        return tsne_components


    '''
    Function to perform UMAP
    '''
    def umap(self):
        umap_emb = umap.UMAP(n_components=self.n_components)
        umap_components = umap_emb.fit_transform(self.df)

        # save learned embedding
        self.umap_emb = umap_emb

        if self.plots[0]:
            self.visualize(umap_components)

        return umap_components


    def test_umap(self, test_df):
        components = self.umap_emb.transform(test_df)
        return components


    '''
    Function to perform Autoencoder Dim Red
    '''
    def autoencoder(self):
        X = self.df.values
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        input_dim = X.shape[1]
        encoding_dim = self.n_components

        # Model building
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        encoder_layer1 = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
        encoder_layer2 = tf.keras.layers.Dense(encoding_dim // 2, activation='relu')(encoder_layer1)
        decoder_layer1 = tf.keras.layers.Dense(encoding_dim // 2, activation='relu')(encoder_layer2)
        decoder_layer2 = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoder_layer1)
        autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder_layer2)
        
        # Compile the model
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        # Train the autoencoder
        history = autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

        # Use the trained autoencoder to perform dimensionality reduction
        encoder = tf.keras.models.Model(inputs=input_layer, outputs=encoder_layer2)
        X_encoded = encoder.predict(X_scaled)

        if self.plots[0]:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            # Assuming X_encoded is the matrix with reduced dimensions
            ax.scatter(X_encoded[:, 0], np.zeros_like(X_encoded[:, 0]), np.zeros_like(X_encoded[:, 0]), s=50)
            ax.set_xlabel('Component 1')
            ax.set_title('Autoencoder Reduced Data Scatter Plot')
            plt.show()
            
        return X_encoded