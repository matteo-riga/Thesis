import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans, OPTICS, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from minisom import MiniSom

class ClusterData():
    def __init__(self, data, plots = [0,0]):
        self.data = data
        self.plots = plots
        self.random_state = 42

    def visualize(self, labels):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        unique_labels = np.unique(labels)
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
        for label, color in zip(unique_labels, colors):
            cluster_data = self.data[labels == label]
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], c=[color], label=f'Cluster {label}')

        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        plt.title('Clustering')
        plt.legend()
        plt.show()

    def hopkins_statistic(self,n_samples=100):
        """
        Calculate the Hopkins statistic for the given dataset.
    
        Parameters:
        - X: The input dataset (array-like or pandas DataFrame).
        - n_samples: The number of points to sample for the Hopkins statistic calculation.
    
        Returns:
        - H: The Hopkins statistic value.
        """
        X = self.data
        d = X.shape[1]
        n = X.shape[0]
        
        if n_samples > n:
            n_samples = n
    
        # Select `n_samples` random data points from X
        np.random.seed(42)
        random_indices = np.random.choice(n, n_samples, replace=False)
        X_sample = X[random_indices]
    
        # Generate `n_samples` random points in the same range as X
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        uniform_random_points = np.random.uniform(min_vals, max_vals, (n_samples, d))
    
        # Fit nearest neighbors model to X
        nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    
        # Calculate nearest neighbor distances for random points in the dataset
        distances_X, _ = nbrs.kneighbors(X_sample)
        distances_X = distances_X[:, 1]
    
        # Calculate nearest neighbor distances for uniformly random points
        distances_random, _ = nbrs.kneighbors(uniform_random_points)
        distances_random = distances_random[:, 0]
    
        # Compute the Hopkins statistic
        H = np.sum(distances_random) / (np.sum(distances_random) + np.sum(distances_X))
    
        return H

    def hopkins(self):
        H = self.hopkins_statistic()
        print(f"Hopkins statistic: {H}")
        return H
        

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
                n_labels = len(set(labels)) - (1 if -1 in labels else 0) # Excluding noise label -1
                if 1 < n_labels < len(labels):  # Ensure more than one cluster is formed
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
            self.visualize(labels)
        
        return labels

    # Optimizer for k-means
    def optimize_k_means(self, max_clusters=10):
        """
        Optimize the number of clusters for K-means using the elbow method.
    
        Parameters:
        - data: Input data (array-like or pandas DataFrame).
        - max_clusters: Maximum number of clusters to test.
    
        Returns:
        - optimal_clusters: Optimal number of clusters determined by the elbow method.
        - elbow_plot: Matplotlib figure showing the elbow plot.
        """
        data = self.data
        inertias = []
        cluster_range = range(1, max_clusters + 1)
    
        for num_clusters in cluster_range:
            kmeans = KMeans(num_clusters, random_state=self.random_state)
            kmeans.fit(data)
            kmeans_labels = kmeans.labels_
            kmeans_centers = kmeans.cluster_centers_
            inertia = kmeans.inertia_
            inertias.append(inertia)

        elbow_point = np.diff(inertias, 2).argmin() + 2  # +2 because the diff reduces the index by 2
    
        # Plot the elbow curve
        if self.plots[1]:
            print("Optimal number of clusters:", elbow_point)
            plt.figure(figsize=(8, 5))
            plt.plot(cluster_range, inertias, 'bx-')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Inertia')
            plt.title('Elbow Method For Optimal Number of Clusters')
            plt.grid(True)
            plt.show()

        return elbow_point
    

    # write the optimizer
    def k_means(self, max_clusters=10):
        """
        Perform K-means clustering on the input data.
    
        Parameters:
        - data: Input data (array-like or pandas DataFrame).
        - num_clusters: Number of clusters to create.
    
        Returns:
        - kmeans_labels: Cluster labels assigned by K-means.
        - kmeans_centers: Coordinates of cluster centers.
        """
        data = self.data
        optimal_clusters = self.optimize_k_means(max_clusters)
        # Fit the model to the data and predict cluster labels
        kmeans = KMeans(optimal_clusters, random_state=self.random_state)
        kmeans.fit(data)
        kmeans_labels = kmeans.labels_
        kmeans_centers = kmeans.cluster_centers_
        inertia = kmeans.inertia_

        # plotting results
        if self.plots[0]:
            self.visualize(kmeans_labels)
    
        return kmeans_labels, kmeans_centers

    # Optimizer for k-medoids
    def optimize_k_medoids(self, max_clusters=10):
        """
        Optimize the number of clusters for K-means using the elbow method.
    
        Parameters:
        - data: Input data (array-like or pandas DataFrame).
        - max_clusters: Maximum number of clusters to test.
    
        Returns:
        - optimal_clusters: Optimal number of clusters determined by the elbow method.
        - elbow_plot: Matplotlib figure showing the elbow plot.
        """
        data = self.data
        inertias = []
        cluster_range = range(1, max_clusters + 1)
    
        for num_clusters in cluster_range:
            kmedoids = KMedoids(n_clusters=num_clusters, random_state=self.random_state)
            kmedoids.fit(data)
            kmedoids_labels = kmedoids.labels_
            kmedoids_centers = kmedoids.cluster_centers_
            inertia = kmedoids.inertia_
            inertias.append(inertia)

        elbow_point = np.diff(inertias, 2).argmin() + 2  # +2 because the diff reduces the index by 2
    
        # Plot the elbow curve
        if self.plots[1]:
            print("Optimal number of clusters:", elbow_point)
            plt.figure(figsize=(8, 5))
            plt.plot(cluster_range, inertias, 'bx-')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Inertia')
            plt.title('Elbow Method For Optimal Number of Clusters')
            plt.grid(True)
            plt.show()

        return elbow_point
    
    # write the optimizer
    def k_medoids(self, max_clusters=10):
        """
        Perform K-medoids clustering on the input data.
    
        Parameters:
        - data: Input data (array-like or pandas DataFrame).
        - num_clusters: Number of clusters to create.
    
        Returns:
        - kmedoids_labels: Cluster labels assigned by K-medoids.
        - kmedoids_medoids: Indices of data points chosen as medoids.
        """
        data = self.data
        # optimizer works also for k medoids
        optimal_clusters = self.optimize_k_medoids(max_clusters)
        # Fit the model to the data and predict cluster labels
        kmedoids = KMedoids(n_clusters=optimal_clusters, random_state=42)
        kmedoids.fit(data)
        kmedoids_labels = kmedoids.labels_
        kmedoids_medoids = kmedoids.cluster_centers_

        # plotting results
        if self.plots[0]:
            self.visualize(kmedoids_labels)
    
        return kmedoids_labels, kmedoids_medoids


    def optimize_optics(self):
        """
        Optimize OPTICS clustering parameters based on silhouette score.
    
        Parameters:
        - data: Input data (array-like or pandas DataFrame).
        - min_samples_range: Range of min_samples to test.
        - xi_range: Range of xi values to test.
        - min_cluster_size_range: Range of min_cluster_size values to test.
    
        Returns:
        - best_params: Best parameters found.
        - best_score: Best silhouette score.
        """
        data = self.data
        best_score = -1  # Initialize with a low value
        best_params = {}
    
        min_samples_range = range(2, 10)
        xi_range = np.linspace(0.01, 0.1, 10)
        min_cluster_size_range = np.linspace(0.05, 0.2, 4)
    
        for min_samples in min_samples_range:
            for xi in xi_range:
                for min_cluster_size in min_cluster_size_range:
                    optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
                    optics.fit(data)
                    labels = optics.labels_
                    n_labels = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise label -1
    
                    if 1 < n_labels < len(labels):  # Ensure more than one cluster is formed
                        score = silhouette_score(data, labels)
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'min_samples': min_samples,
                                'xi': xi,
                                'min_cluster_size': min_cluster_size
                            }

        return best_params, best_score


    # write the optimizer
    def optics(self):
        """
        Perform OPTICS clustering on the input data.
    
        Parameters:
        - data: Input data (array-like or pandas DataFrame).
        - min_samples: The number of samples in a neighborhood for a data point to be considered as a core point.
        - xi: Determines the minimum steepness of the cluster hierarchy. Smaller values result in finer cluster hierarchies.
        - min_cluster_size: Minimum number of samples in a cluster.
    
        Returns:
        - optics_labels: Cluster labels assigned by OPTICS.
        """
        data=self.data

        best_params, best_score = self.optimize_optics()
        
        # Create OPTICS model
        optics = OPTICS(min_samples=best_params['min_samples'], xi=best_params['xi'], min_cluster_size=best_params['min_cluster_size'])
    
        # Fit the model to the data and predict cluster labels
        optics.fit(data)
        optics_labels = optics.labels_

        # plotting results
        if self.plots[0]:
            self.visualize(optics_labels)

        return optics_labels



    def optimize_hierarchical(self, max_clusters=10):
        """
        Optimize the number of clusters for hierarchical clustering using the silhouette score.
    
        Parameters:
        - max_clusters: Maximum number of clusters to test.
    
        Returns:
        - best_n_clusters: Best number of clusters.
        - best_score: Best silhouette score.
        """
        best_score = -1  # Initialize with a low value
        best_n_clusters = 2  # Minimum number of clusters to test
    
        for n_clusters in range(2, max_clusters + 1):
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(self.data)
            score = silhouette_score(self.data, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
    
        return best_n_clusters, best_score


    def hierarchical(self, max_clusters=10):
        """
        Perform hierarchical clustering on the input data with optimized number of clusters.

        Returns:
        - labels: Cluster labels assigned by hierarchical clustering.
        """
        best_n_clusters, best_score = self.optimize_hierarchical()

        # Perform hierarchical clustering with the best number of clusters
        clustering = AgglomerativeClustering(n_clusters=best_n_clusters)
        labels = clustering.fit_predict(self.data)
        # Plotting results if required
        if self.plots[0]:
            self.visualize(labels)
        
        return labels


    def som_clustering(self, x_dim, y_dim, sigma, learning_rate):
        """
        Perform SOM clustering on the input data.

        Parameters:
        - x_dim: x dimension of the SOM grid.
        - y_dim: y dimension of the SOM grid.
        - sigma: Spread of the neighborhood function.
        - learning_rate: Learning rate of the SOM.

        Returns:
        - labels: Cluster labels assigned by SOM.
        """
        som = MiniSom(x_dim, y_dim, self.data.shape[1], sigma=sigma, learning_rate=learning_rate)
        som.random_weights_init(self.data)
        som.train_random(self.data, 1000)

        # Assign each data point to the nearest neuron
        labels = np.array([som.winner(x) for x in self.data])
        labels = labels[:, 0] * y_dim + labels[:, 1]  # Convert 2D coordinates to 1D label

        return labels

    def optimize_som(self, max_x_dim=10, max_y_dim=10, sigma_range=(0.5, 1.5), learning_rate_range=(0.1, 0.5)):
        """
        Optimize the SOM parameters using the silhouette score.

        Parameters:
        - max_x_dim: Maximum x dimension of the SOM grid.
        - max_y_dim: Maximum y dimension of the SOM grid.
        - sigma_range: Range of sigma values to test.
        - learning_rate_range: Range of learning rate values to test.

        Returns:
        - best_params: Best parameters found.
        - best_score: Best silhouette score.
        """
        best_score = -1  # Initialize with a low value
        best_params = {}

        for x_dim in range(2, max_x_dim + 1):
            for y_dim in range(2, max_y_dim + 1):
                for sigma in np.linspace(*sigma_range, num=5):
                    for learning_rate in np.linspace(*learning_rate_range, num=5):
                        labels = self.som_clustering(x_dim, y_dim, sigma, learning_rate)
                        score = silhouette_score(self.data, labels)
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'x_dim': x_dim,
                                'y_dim': y_dim,
                                'sigma': sigma,
                                'learning_rate': learning_rate
                            }

        return best_params, best_score

    def som(self):
        """
        Perform SOM clustering on the input data with optimized parameters.

        Returns:
        - labels: Cluster labels assigned by SOM.
        """
        best_params, best_score = self.optimize_som()

        # Perform SOM clustering with the best parameters
        labels = self.som_clustering(best_params['x_dim'], best_params['y_dim'], best_params['sigma'], best_params['learning_rate'])
        
        # Plotting results if required
        if self.plots[0]:
            self.visualize(labels)
        
        return labels