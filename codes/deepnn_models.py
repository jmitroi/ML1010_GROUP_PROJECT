# Modeling related
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from nltk.tokenize.toktok import ToktokTokenizer

class CNN():
    def __init__(self,
                 #train_seq_x, valid_seq_x,
                 #train_encoded_y, valid_encoded_y,
                 num_unique_word_in_corpus, embedding_matrix,
                 input_vector_len, num_classes):
        """
        :param train_seq_x,valid_seq_x: text data represented by word vectors
        :param train_encoded_y,valid_encoded_y: one hot encoded labels
        :param num_classes: number of classes in labels, used to set the last layers of CNN
        :param input_vector_len: used to set the dimensions of first layers in CNN
        :param num_unique_word_in_corpus: the number of unique words seen in the given corpus
        :param embedding_matrix: word index -> word vector mapping
        """
        # self.train_seq_x = train_seq_x
        # self.valid_seq_x = valid_seq_x
        # self.train_encoded_y = train_encoded_y
        # self.valid_encoded_y = valid_encoded_y
        self.input_vector_len = input_vector_len
        self.num_classes = num_classes
        self.num_unique_word_in_corpus = num_unique_word_in_corpus
        self.embedding_matrix = embedding_matrix

    def create_model(self):
        input_layer = layers.Input((self.input_vector_len,))
        embedding_layer = layers.Embedding(self.num_unique_word_in_corpus, 300, weights=[self.embedding_matrix], trainable=False)(
            input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
        conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)
        pooling_layer = layers.GlobalMaxPool1D()(conv_layer)
        output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(self.num_classes, activation="sigmoid")(output_layer1)
        final_layer = layers.Activation('softmax', name="softmax")(output_layer2)
        # Compile the model
        model = models.Model(inputs=input_layer, outputs=final_layer)
        model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=["accuracy"])
        self.model = model
        return model
