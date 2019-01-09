from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.preprocessing import text, sequence
import numpy as np
import pickle

class VectorizerBase:
    def __init__(self):
        self.vec = None

    def save(self, file_path):
        print("saving vectorizer to " + file_path)
        with open(file_path, 'wb') as handle:
            pickle.dump(self.vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        print("loading vectorizer from " + file_path)
        with open(file_path, 'rb') as handle:
            self.vec = pickle.load(handle)

class VectorizerTFIDF(VectorizerBase):
    def __init__(self, max_features=None, ngram_range=(1, 2)):
        super().__init__()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vec = TfidfVectorizer(min_df=3, max_features=self.max_features, \
                                   token_pattern=r'\w{1,}',
                                   ngram_range=self.ngram_range)

    def clone(self):
        self.vec = TfidfVectorizer(min_df=3, max_features=self.max_features, \
                                   token_pattern=r'\w{1,}',
                                   ngram_range=self.ngram_range)
        return self

    def fit(self, X, y=None):
        # y not used
        self.vec.fit(X)
        pass

    def transform(self,X):
        return self.vec.transform(X)

class VectorizerCountVec(VectorizerBase):
    def __init__(self, max_features=None, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vec = CountVectorizer(token_pattern=r'\w{1,}',max_features=self.max_features,
                                   ngram_range=self.ngram_range)

    def clone(self):
        self.vec = CountVectorizer(token_pattern=r'\w{1,}',max_features=self.max_features,
                                   ngram_range=self.ngram_range)
        return self

    def fit(self, X, y=None):
        # y not used
        self.vec.fit(X)
        pass

    def transform(self,X):
        return self.vec.transform(X)

class VectorizerCountVecNB(VectorizerBase):
    def __init__(self, max_features=None, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vec = CountVectorizer(token_pattern=r'\w{1,}',max_features=self.max_features,
                                   ngram_range=self.ngram_range)
        self.r = None

    def clone(self):
        self.vec = CountVectorizer(token_pattern=r'\w{1,}',max_features=self.max_features,
                                   ngram_range=self.ngram_range)
        self.r = None
        return self

    def pr(self, x, y_i, y):
        p = x[y == y_i].sum(0)
        return (p + 1) / ((y == y_i).sum() + 1)

    def fit(self, X, y):
        self.vec.fit(X)
        X = self.vec.transform(X)
        y = y.squeeze()
        self.r = np.log(self.pr(X, 1, y) / self.pr(X, 0, y))

    def transform(self,X):
        return self.vec.transform(X).multiply(self.r)

    def save(self, file_path):
        print("saving vectorizer to " + file_path)
        to_save = {"vec":self.vec, "r":self.r}
        with open(file_path, 'wb') as handle:
            pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        print("loading vectorizer from " + file_path)
        with open(file_path, 'rb') as handle:
            from_save = pickle.load(handle)
            self.vec = from_save["vec"]
            self.r = from_save["r"]

class VectorizerTFIDFNB(VectorizerBase):
    def __init__(self, max_features=None, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vec = TfidfVectorizer(min_df=3, max_features=self.max_features, \
                                   token_pattern=r'\w{1,}',
                                   ngram_range=self.ngram_range)
        self.r = None

    def clone(self):
        self.vec = TfidfVectorizer(min_df=3, max_features=self.max_features, \
                                   token_pattern=r'\w{1,}',
                                   ngram_range=self.ngram_range)
        self.r = None
        return self

    def pr(self, x, y_i, y):
        p = x[y == y_i].sum(0)
        return (p + 1) / ((y == y_i).sum() + 1)

    def fit(self, X, y):
        self.vec.fit(X)
        X = self.vec.transform(X)
        y = y.squeeze()
        self.r = np.log(self.pr(X, 1, y) / self.pr(X, 0, y))

    def transform(self,X):
        return self.vec.transform(X).multiply(self.r)

    def save(self, file_path):
        print("saving vectorizer to " + file_path)
        to_save = {"vec":self.vec, "r":self.r}
        with open(file_path, 'wb') as handle:
            pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        print("loading vectorizer from " + file_path)
        with open(file_path, 'rb') as handle:
            from_save = pickle.load(handle)
            self.vec = from_save["vec"]
            self.r = from_save["r"]

class VectorizerEmbedding:
    def __init__(self, word_vector_file='../wordvecs/wiki-news-300d-1M.vec',
                 docLen=5000):
        """
        :param embed_size: size of embedding vector. For fasttext, it is 300
        :param docLen: max number of words in a document to use. If document has less words, padding is used. If None,
        use max doc length in training set
        """
        self.word_vector_file = word_vector_file
        self.embeddingMatrix = None
        self.tokenizer = None
        self.docLen = docLen

    def clone(self):
        self.embeddingMatrix = None
        self.tokenizer = None
        return self

    def fit(self, X, y=None):
        # y not used
        print("Building embedding matrix.")
        self.tokenizer = text.Tokenizer()
        self.tokenizer.fit_on_texts(X)
        embeddings_index = {}
        embed_size = None
        for i, line in enumerate(open(self.word_vector_file)):
            values = line.split()
            if len(values[1:]) <= 2:
                continue
            embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
        embed_size = len(next(iter(embeddings_index.values())))
        print("Total words in embedding file:" + str(len(embeddings_index)))
        word_index = self.tokenizer.word_index
        nb_words = len(word_index)
        # for words not in embedding file, init them to a random values
        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words + 1, embed_size))
        for word, i in sorted(word_index.items(), key=lambda kv: kv[1]):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        self.embeddingMatrix = embedding_matrix
        print("Build embedding matrix completes.")
        return

    def transform(self, X):
        # return tokenized representation of texts (X)
        if self.tokenizer is not None:
            text_tokens = self.tokenizer.texts_to_sequences(X)
            if self.docLen is None:
                max_tokens_one_sent = 0
                for doc in text_tokens:
                    max_tokens_one_sent = max(max_tokens_one_sent, len(doc))
                self.docLen = max_tokens_one_sent
            text_seqences = sequence.pad_sequences(text_tokens, maxlen=self.docLen)
            return text_seqences
        else:
            raise ValueError

    def save(self, file_path):
        to_save = {"tokenizer": self.tokenizer,
                   "docLen": self.docLen,
                   "embeddingMatrix": self.embeddingMatrix}
        print("saving vectorizer to " + file_path)
        with open(file_path, 'wb') as handle:
            pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        print("loading vectorizer from " + file_path)
        with open(file_path, 'rb') as handle:
            from_saved = pickle.load(handle)
        self.tokenizer = from_saved["tokenizer"]
        self.docLen = from_saved["docLen"]
        self.embeddingMatrix = from_saved["embeddingMatrix"]

