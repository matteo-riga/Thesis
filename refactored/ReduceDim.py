import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

class ReduceDim():
    def __init__(self, n_components, df, plots = [0,0]):
        self.n_components = n_components
        self.df = df
        self.plots = plots

    def pca(self):
        pca = PCA()
        pca.fit(self.df)
        transformed_data = pca.transform(self.df)

        if self.plots[0]:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], s=50)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            plt.title('Scatter Plot along the first 3 PCs')
            plt.show()

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

    def tsne(self):
        tsne = TSNE(n_components=self.n_components)
        tsne_components = tsne.fit_transform(self.df)

        if self.plots[0]:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(tsne_components[:, 0], tsne_components[:, 1], tsne_components[:, 2], s=50)
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.set_zlabel('t-SNE Component 3')
            plt.title('Scatter Plot along t-SNE Components')
            plt.show()
        
        return tsne_components

    def umap(self):
        umap_emb = umap.UMAP(n_components=self.n_components)
        umap_components = umap_emb.fit_transform(self.df)

        if self.plots[0]:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(umap_components[:, 0], umap_components[:, 1], umap_components[:, 2], s=50)
            ax.set_xlabel('UMAP Component 1')
            ax.set_ylabel('UMAP Component 2')
            ax.set_zlabel('UMAP Component 3')
            plt.title('Scatter Plot along UMAP Components')
            plt.show()
        return umap_components