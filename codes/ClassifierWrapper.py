from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from keras import layers, models, optimizers
from keras.callbacks import EarlyStopping
import pickle
from keras.models import load_model

class ClassifierWrapperBase():
    def __init__(self):
        self.useEmbedding = False
        self.useValEarlyStop = False
        self.isKerasClf = False
        self.history = []

class LogisticRegressionWrapper(ClassifierWrapperBase):
    def __init__(self,
                 penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='warn', max_iter=100,
                 multi_class='warn', verbose=0, warm_start=False, n_jobs=None
                 ):
        super().__init__()
        self.model = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C,
                                        fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                        class_weight=class_weight, random_state=random_state, solver=solver,
                                        max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start,
                                        n_jobs=n_jobs)

    def fit(self, X, y, validation_data=None, epochs=None):
        self.model.fit(X,y)
        return None

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]

    def clone(self):
        self.model = clone(self.model)
        return self

    def save(self, file_path):
        print("saving model to " + file_path)
        with open(file_path, 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        print("loading model from " + file_path)
        with open(file_path, 'rb') as handle:
            self.model = pickle.load(handle)


class CNNWrapper(ClassifierWrapperBase):
    def __init__(self, docLen=5000):
        """
        :param docLen: max number of words in a document to use. If document has less words, padding is used.
        """
        super().__init__()
        self.docLen = docLen
        self.numUniqueWord = None
        self.embdeddingMatrix = None
        self.train_epochs = 200
        self.useEmbedding = True
        self.useValEarlyStop = True
        self.isKerasClf = True
        self.model = None
        self.history = [] #history is not reset when cloned

    def fit(self, X, y, validation_data=None, epochs=None):
        # validation_data=(X_test,y_test)
        if validation_data is not None:
            early = EarlyStopping(monitor="acc", mode="max", patience=5)
            callbacks_list = [early]
            history = self.model.fit(x=X, y=y, epochs=self.train_epochs,
                                     validation_data=validation_data,
                                     callbacks=callbacks_list)
        else:
            if epochs is None:
                epochs = 10
            history = self.model.fit(x=X, y=y, epochs=epochs)
        return history

    def predict_proba(self, X):
        return self.model.predict(X)

    def clone(self):
        input_layer = layers.Input((self.docLen,))
        embedding_layer = layers.Embedding(self.numUniqueWord + 1, len(self.embdeddingMatrix[0]),
                                           weights=[self.embdeddingMatrix], trainable=False)(input_layer)
        embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
        conv_layer = layers.Convolution1D(100, 5, activation="relu")(embedding_layer)
        pooling_layer = layers.GlobalMaxPool1D()(conv_layer)
        output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
        output_layer1 = layers.Dropout(0.25)(output_layer1)
        output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)
        # Compile the model
        model = models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=["accuracy"])
        self.model = model
        return self

    def save(self, file_path):
        print("saving models to " + file_path)
        self.model.save(file_path)

    def load(self, file_path):
        print("loading model from " + file_path)
        self.model = load_model(file_path)
