from transformers import BertTokenizer, BertModel
import torch
import numpy as np
#from gensim.models import KeyedVectors

class SentenceEmbedding():
    def __init__(self):
        # Load pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    '''
    def get_average_word2vec_embedding(self, tokens):
        self.word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        embeddings = []
        for token in tokens:
            if token in self.word2vec_model:
                embeddings.append(self.word2vec_model[token])
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(self.word2vec_model.vector_size)
    '''

    def get_bert_embedding(self, text):
        # Tokenize text and convert to PyTorch tensors
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the [CLS] token embedding as the sentence embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return cls_embedding