import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

# Get encoded df after preprocessing
# encoded_df = one_hot_encoding.one_hot_encode_after_preprocessing()


def perform_PCA(encoded_df, plot_points, plot_cum_var):
    # PCA
    for i,df in enumerate(encoded_df):
        pca = PCA()
        pca.fit(df)
        transformed_data = pca.transform(df)

        if plot_points:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], s=50)

            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')

            plt.title(f'Scatter Plot along the first 3 PCs for df {i}')
            plt.show()
        
        if plot_cum_var:
            # Plot the cumulative explained variance
            plt.subplot(1, 2, 2)
            plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='-')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title(f'Cumulative Explained Variance df {i}')
            plt.grid(True)

            plt.tight_layout()
            plt.show()

def perform_UMAP(encoded_df, plot_points=False):
    for i,df in enumerate(encoded_df):
        # Perform UMAP decomposition
        umap_emb = umap.UMAP(n_components=3)  # Set the number of components to 3 for 3D visualization
        umap_components = umap_emb.fit_transform(df)

        if plot_points:
            # Plotting scatter plot along the UMAP components
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(umap_components[:, 0], umap_components[:, 1], umap_components[:, 2], s=50)

            ax.set_xlabel('UMAP Component 1')
            ax.set_ylabel('UMAP Component 2')
            ax.set_zlabel('UMAP Component 3')

            plt.title(f'Scatter Plot along UMAP Components for dataframe {i}')
            plt.show()
