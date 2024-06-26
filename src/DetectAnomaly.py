import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

class DetectAnomaly:
    def __init__(self, data, plots=[0, 0], random_state=42):
        """
        Initialize the DetectAnomaly class.

        Parameters:
        - data: Input data (array-like or pandas DataFrame).
        - plots: List indicating whether to generate plots for each method (0 for no plot, 1 for plot).
        - random_state: Random state for reproducibility.
        """
        self.data = data
        self.plots = plots
        self.random_state = random_state

        
    def visualize(self, anomalies):
        """
        Visualize the data points with anomalies highlighted.
    
        Parameters:
        - anomalies: Data points classified as anomalies.
    
        Returns:
        - None (plots the visualization).
        """
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
    
        # Create boolean index for anomalies
        is_anomaly = np.zeros(len(self.data), dtype=bool)
        is_anomaly[anomalies] = True
    
        # Plot normal data points
        normal_data = self.data[~is_anomaly]
        ax.scatter(normal_data[:, 0], normal_data[:, 1], normal_data[:, 2], c='blue', label='Normal')
    
        # Plot anomalous data points
        ax.scatter(self.data[is_anomaly][:, 0], self.data[is_anomaly][:, 1], self.data[is_anomaly][:, 2], c='red', label='Anomaly')
    
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        plt.title('Clustering with Anomalies Highlighted')
        plt.legend()
        plt.show()


    def find_anomalies(self, anomaly_scores):
        """
        Find anomalies based on anomaly scores and a given threshold.
    
        Parameters:
        - anomaly_scores: Anomaly scores calculated by the anomaly detection method.
        - threshold: Threshold above which data points are considered anomalies.
    
        Returns:
        - anomaly_indices: Indices of data points classified as anomalies.
        - anomaly_data_points: Data points classified as anomalies.
        """
        
        # Normalize anomaly scores between 0 and 1
        #min_score = np.min(anomaly_scores)
        #max_score = np.max(anomaly_scores)
        #normalized_scores = (anomaly_scores - min_score) / (max_score - min_score)

        mean_score = np.mean(anomaly_scores)
        std_dev_score = np.std(anomaly_scores)
        normalized_scores = (anomaly_scores - mean_score) / std_dev_score

        # Choose to classify as anomaly everything exceeding the 3 sigma
        threshold = 3 * np.std(normalized_scores)
        
        # Find indices where anomaly scores exceed the threshold
        anomaly_indices = np.where(normalized_scores > threshold)[0]
        
        # Get data points corresponding to anomaly indices
        anomaly_data_points = self.data[anomaly_indices]

        if self.plots[1]:
            print("Anomaly indices:", anomaly_indices)
            print("Anomaly data points:", anomaly_data_points)
        if self.plots[0]:
            self.visualize(anomaly_indices)
            plt.figure(figsize=(8, 6))
            plt.scatter(np.arange(len(anomaly_scores)), normalized_scores, c='red', alpha=0.5)
            plt.xlabel('Data Point Index')
            plt.ylabel('Normalized Anomaly Score')
            plt.title('Normalized Anomaly Scores')
            plt.grid(True)
            plt.axhline(y=threshold, color='blue', linestyle='--')
            plt.show()
        
        return anomaly_indices, anomaly_data_points


    def knn(self, n_neighbors=5):
        """
        Perform k-Nearest Neighbors (KNN) anomaly detection using the PyOD library.
    
        Parameters:
        - data: Input data (array-like or pandas DataFrame).
        - n_neighbors: Number of neighbors to consider for anomaly detection (default is 5).
    
        Returns:
        - anomaly_scores: Anomaly scores calculated by the KNN method.
        """
        # Initialize KNN model
        knn_model = KNN(n_neighbors=n_neighbors)
        
        # Fit the model to the data and obtain anomaly scores
        knn_model.fit(self.data)
        anomaly_scores = knn_model.decision_scores_
    
        return anomaly_scores


    def lof(self, n_neighbors=20):
        """
        Perform Local Outlier Factor (LOF) anomaly detection using the PyOD library.
    
        Parameters:
        - data: Input data (array-like or pandas DataFrame).
        - n_neighbors: Number of neighbors to consider for LOF calculation (default is 20).
    
        Returns:
        - anomaly_scores: Anomaly scores calculated by the LOF method.
        """
        # Initialize LOF model
        lof_model = LOF(n_neighbors=n_neighbors)
        
        # Fit the model to the data and obtain anomaly scores
        lof_model.fit(self.data)
        anomaly_scores = lof_model.decision_scores_
    
        return anomaly_scores

    def isolation_forest(self):
        """
        Perform Isolation Forest anomaly detection using the PyOD library.
    
        Parameters:
        - data: Input data (array-like or pandas DataFrame).
    
        Returns:
        - anomaly_scores: Anomaly scores calculated by the Isolation Forest method.
        """
        # Initialize Isolation Forest model
        if_model = IForest(random_state=42)
        
        # Fit the model to the data and obtain anomaly scores
        if_model.fit(self.data)
        anomaly_scores = if_model.decision_scores_
    
        return anomaly_scores

    def mcd(self):
        """
        Perform Minimum Covariance Determinant (MCD) anomaly detection.

        Returns:
        - anomaly_scores: Anomaly scores calculated by the MCD method.
        """
        pass

    def hidden_markov_models(self):
        """
        Perform Hidden Markov Models (HMM) anomaly detection.

        Returns:
        - anomaly_scores: Anomaly scores calculated by the HMM method.
        """
        pass
