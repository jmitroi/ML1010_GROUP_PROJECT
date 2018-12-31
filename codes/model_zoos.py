from keras import layers, models, optimizers


class CnnWrapper:
    def __init__(self,
                 embedding_matrix,
                 max_features,
                 max_words):
        """
        :param embedding_matrix: word index -> word vector mapping
        :param max_features: how many unique words to use (i.e num rows in embedding vector)
        :param max_words = 100 # max number of words in a document to use

        """
        self.max_features = max_features
        self.max_words = max_words
        self.embedding_matrix = embedding_matrix
        self.model = None

    def create_model(self):
        input_layer = layers.Input((self.max_words,))
        embedding_layer = layers.Embedding(self.max_features+1, len(self.embedding_matrix[0]),
                                           weights=[self.embedding_matrix], trainable=False)(input_layer)
        embedding_layer = layers.SpatialDropout1D(0.5)(embedding_layer)
        conv_layer = layers.Convolution1D(100, 5, activation="relu")(embedding_layer)
        pooling_layer = layers.GlobalMaxPool1D()(conv_layer)
        output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
        output_layer1 = layers.Dropout(0.5)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)
        # Compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=["accuracy"])
        print(model.summary())
        self.model = model
        return self.model


class LstmWrapper:
    def __init__(self,
                 embedding_matrix,
                 max_features,
                 max_words):
        """
        :param embedding_matrix: word index -> word vector mapping
        :param max_features: how many unique words to use (i.e num rows in embedding vector)
        :param max_words = 100 # max number of words in a document to use

        """
        self.max_features = max_features
        self.max_words = max_words
        self.embedding_matrix = embedding_matrix
        self.model = None

    def create_model(self):
        inp = layers.Input(shape=(self.max_words,))
        x = layers.Embedding(self.max_features+1, len(self.embedding_matrix[0]),
                             weights=[self.embedding_matrix])(inp)
        x = layers.Bidirectional(layers.LSTM(50, return_sequences=True,
                                             dropout=0.1, recurrent_dropout=0.1))(x)
        x = layers.GlobalMaxPool1D()(x)
        x = layers.Dense(50, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(1, activation="sigmoid")(x)
        model = models.Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        self.model = model
        return self.model