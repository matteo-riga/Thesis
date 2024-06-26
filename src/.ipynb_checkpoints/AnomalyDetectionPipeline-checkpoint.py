class AnomalyDetectionPipeline():

    def __init__(self):
        pass


    def umap(self, train=0):
        # train parametric umap (if train = 1)
        # get data point with parametric umap
        pass
        

    def dbscan(self, train):
        # from the umap decomposed data point
        # return the closest cluster
        pass
        

    def autoencoder(self, train = 0):
        # train autoencoder with fixed hyperparameters
        # return autoencoder prediction
        pass


    def LSTM(self, train):
        # train LSTM
        # predict with LSTM
        pass


    def transformer(self, train):
        pass

    
    def train(self, df):
        # train all the trainable models
        # save models to file
        pass


    def test(self, df):
        # return the tests for all the models
        pass