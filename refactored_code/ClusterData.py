import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

class ClusterData():
    def __init__(self, data, plots = [0]):
        self.data = data

    def optimize_dbscan(self):
        data = self.data

        best_eps = None
        best_min_samples = None
        best_score = -1  # Initialize with a low value

        n_samples = len(data)
        eps_range = np.linspace(0.1, 2.0, num=20) 
        min_samples_range = np.linspace(1, 20, num=5, dtype=int)
        #min_samples_range = [min(5, n_samples // 10), min(10, n_samples // 5)]  # Adjust as needed

        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(data)
                if len(set(labels)) > 1:  # Ensure more than one cluster is formed
                    score = silhouette_score(data, labels)
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_min_samples = min_samples

        self.best_eps = best_eps
        self.best_min_samples = best_min_samples
        self.best_score = best_score

        return best_eps, best_min_samples

    def dbscan(self):
        
        # Optimization of dbscan parameters
        eps, min_samples = self.optimize_dbscan()
        # Fitting dbscan with optimal values
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.data)

        # plotting results
        if self.plots[0]:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            unique_labels = np.unique(labels)
            colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
            for label, color in zip(unique_labels, colors):
                cluster_data = umap_components[labels == label]
                ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], c=[color], label=f'Cluster {label}')

            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            plt.title('DBSCAN Clustering')
            plt.legend()
            plt.show()
        
        return labels

    def knn():
        pass

    def k_medoids():
        pass