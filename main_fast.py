# numpy_text.py package
import fasttext
import numpy as np
from tqdm import tqdm

class NumpyText:
    def __init__(self, ft_model_path):
        """
        Initialize the NumpyText class.

        Parameters:
        ft_model_path (str): Path to the FastText model.
        """
        self.ft_model = fasttext.load_model(ft_model_path)
        self.encoder = None
        self.model = None
        self.vocabulary = None

    def fit(self, vocabulary):
        """
        Fit the model by creating an array of sentence vectors for each token in the vocabulary.
        """
        self.vocabulary = vocabulary
        self.model = np.array([self.ft_model.get_sentence_vector(token) for token in tqdm(self.vocabulary, position=0, total=len(self.vocabulary))])
        self.encoder = {token: i for i, token in enumerate(vocabulary)}
        self.ft_model = None  # Free up memory

    def predict(self, query):
        """
        Predict the sentence vector for a given query.

        Parameters:
        query (str): The input query.

        Returns:
        np.array: The mean sentence vector for the query.
        """
        tokens = query.split()
        token_ids = [self.encoder[token] for token in tokens if token in self.encoder]

        if not token_ids:
            raise ValueError(f"No tokens in the '{query}' are present in the vocabulary.")

        return np.mean(self.model[token_ids], axis=0)



if __name__ == "__main__":
    # Example usage:
    vocabulary = ["iphone", "15", "pro", "max", "платье", "женское"]
    ft_model_path = "/home/fkurushin/personal-recommendations/ft_sg_st_154178937_dim_16.bin"
    numpy_text = NumpyText(vocabulary, ft_model_path)
    numpy_text.fit()
    query = "iphone 15"
    result = numpy_text.predict(query)
    print(result)
