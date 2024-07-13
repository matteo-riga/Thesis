import numpy as np
from gensim.models import KeyedVectors

class SentenceEmbedding:
    def __init__(self, model_path):
        # Load pre-trained Word2Vec model (Google News vectors)
        self.word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    
    def get_average_word2vec_embedding(self, tokens):
        embeddings = []
        for token in tokens:
            if token in self.word2vec_model:
                embeddings.append(self.word2vec_model[token])
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(self.word2vec_model.vector_size)

# Example usage
if __name__ == "__main__":
    model_path = 'GoogleNews-vectors-negative300.bin'
    sentence_embedding = SentenceEmbedding(model_path)

    tokens = ['this', 'is', 'an', 'example']
    embedding = sentence_embedding.get_average_word2vec_embedding(tokens)
    print(embedding)
