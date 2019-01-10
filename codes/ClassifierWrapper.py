from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from keras import layers, models, optimizers
from keras.callbacks import EarlyStopping
import pickle
from keras.models import load_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier


class ClassifierWrapperBase():
    def __init__(self):
        self.useEmbedding = False
        self.useValEarlyStop = False
        self.isKerasClf = False
        self.history = []
        self.model = None

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


class LogisticRegressionWrapper(ClassifierWrapperBase):
    def __init__(self,
                 penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=800,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=None
                 ):
        super().__init__()
        self.model = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C,
                                        fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                        class_weight=class_weight, random_state=random_state, solver=solver,
                                        max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start,
                                        n_jobs=n_jobs)


class MultinomialNBWrapper(ClassifierWrapperBase):
    def __init__(self,
                 alpha=1.0, fit_prior=True, class_prior=None
                 ):
        super().__init__()
        self.model = MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)


class RandomForestClassifierWrapper(ClassifierWrapperBase):
    def __init__(self,
                 n_estimators='warn',
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None
                 ):
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight
        )


class SVCWrapper(ClassifierWrapperBase):
    def __init__(self,
                 C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None
                 ):
        super().__init__()
        self.model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,
                         coef0=coef0, shrinking=shrinking, probability=probability,
                         tol=tol, cache_size=cache_size, class_weight=class_weight,
                         verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
                         random_state=random_state)


class BaggingClassifierWrapper(ClassifierWrapperBase):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0
                 ):
        super().__init__()
        self.model = BaggingClassifier(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )

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
