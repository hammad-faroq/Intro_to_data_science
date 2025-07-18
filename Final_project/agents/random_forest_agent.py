import joblib
import numpy as np
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
from agents.agent import Agent

class RandomForestAgent(Agent):
    name = "Random Forest Agent"
    color = Agent.MAGENTA

    def __init__(self, rf_model_path="rf_model_w2v.pkl", w2v_path="embeddings.txt"):
        """
        Initialize by loading Random Forest model and Word2Vec embeddings
        """
        self.log("Random Forest Agent is initializing")

        # Load trained Random Forest model
        self.model = joblib.load(rf_model_path)

        # Load Word2Vec embeddings
        self.w2v_vectors = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
        self.vector_size = self.w2v_vectors.vector_size

        self.log("Random Forest Agent is ready")

    def document_vector(self, text: str) -> np.ndarray:
        """
        Convert text into an average Word2Vec vector
        """
        words = simple_preprocess(text)
        vectors = [self.w2v_vectors[word] for word in words if word in self.w2v_vectors]

        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.vector_size)

    def price(self, description: str) -> float:
        """
        Predict price using Random Forest model
        """
        self.log("Random Forest Agent is starting a prediction")

        # Convert description into vector
        doc_vec = self.document_vector(description).reshape(1, -1)

        # Predict price
        predicted_price = max(0, self.model.predict(doc_vec)[0])

        self.log(f"Random Forest Agent completed - predicting ${predicted_price:.2f}")
        return predicted_price
