# numpy-text package
# based on 
# https://github.com/facebookresearch/fastText/blob/main/python/doc/examples/FastTextEmbeddingBag.py - the bad one example does not work thrown an err
# http://christopher5106.github.io/deep/learning/2020/04/02/fasttext_pretrained_embeddings_subword_word_representations.html
# https://github.com/piskvorky/gensim/issues/2059
import fasttext
import warnings
import numpy as np
from tqdm import tqdm

# numpy==1.26.4
warnings.filterwarnings('ignore')

EOS = "</s>"
BOW = "<"
EOW = ">"

def get_hash(string, bucket, nb_words):
    bytes = string.encode('utf-8')
    h = np.uint32(2166136261)
    for b in bytes:
        h = np.uint32(h ^ np.uint32(np.int8(b)))
        h = np.uint32(h * np.uint32(16777619))
    return h % bucket + nb_words

def tokenize(query):
    # original fasttext supports split by
    # ' ', '\n', '\r', '\t', '\v', '\f', '\0'
    # and replace by
    # '\n' - EOS
    return query.split()
    
class NumpyText:
    def __init__(self, ft_model_path, bucket=2000000, minn=3, maxn=6):
        ft_model = fasttext.load_model(ft_model_path)
        self.bucket = bucket
        self.model = ft_model.get_input_matrix()
        self.word2ind = {token : i for i, token in enumerate(ft_model.get_words())}
        self.dim = ft_model.get_dimension()
        self.nb_words = len(self.word2ind)
        self.minn=minn
        self.maxn=maxn
        
        print('Convert FastText to NumpyText model')
        print('params:')
        print(f'bucket size: {self.bucket}')
        print(f'bucket size: {self.nb_words}')
        

    def get_subwords(self, word):
        word_ = BOW + word + EOW
        subwords = []
        subword_ids = []
        if word in self.word2ind:
            subwords.append(word)
            subword_ids.append(self.word2ind[word])
        if word == EOS:
            return subwords, np.array(subword_ids)
        n = len(word_)
        for i in range(n):
            for length in range(self.minn, self.maxn + 1):
                if i + length <= n:
                    subw = word_[i:i + length]
                    subword_ids.append(get_hash(subw, self.bucket, self.nb_words))
                    subwords.append(subw)
        return np.array(subwords), np.array(subword_ids)

    def get_word_vector(self, word):
        word_subinds = np.empty([0], dtype=np.int64)
        _, subinds = self.get_subwords(word)
        return np.mean(self.model[subinds], axis=0)

    def get_sentence_vector(self, query):
        tokens = tokenize(query)
        vectors = []
        for t in tokens:
            vec = self.get_word_vector(t)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.mean(vectors, axis=0)




if __name__ == "__main__":
    # Example usage:
    ft_model_path = "/home/fkurushin/personal-recommendations/ft_sg_st_154178937_dim_16.bin"
    ft_model = fasttext.load_model(ft_model_path)
    numpy_text = NumpyText(ft_model_path, 3000000)
    
    query = "чехол iphone 15 pro max"
    print("mine:", numpy_text.get_sentence_vector(query))
    print("ft:", ft_model.get_sentence_vector(query))
    # print(numpy_text.get_subwords(query)[1])
    # print(ft_model.get_subwords(query)[1])
   

    
