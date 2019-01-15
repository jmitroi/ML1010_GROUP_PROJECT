from sklearn.model_selection import StratifiedKFold
from sklearn import metrics,preprocessing
import pandas as pd
import re
import numpy as np
import os
import glob
import pickle


class TextClassifier:
    def __init__(self, vectorizerList, classifierList):
        # classifierList must be from ClassifierWrapper
        self.classifierList = classifierList
        self.vectorizerList = vectorizerList
        assert type(classifierList) is list
        assert type(vectorizerList) is list
        assert len(classifierList) == len(vectorizerList)

    def fit(self, X, y):
        label_encoder = preprocessing.LabelBinarizer()
        y = label_encoder.fit_transform(y)
        X = self.additional_preprocess(X)
        numClfs = len(self.classifierList)
        for k in range(numClfs):
            # Train vectorizerList
            self.vectorizerList[k] = self.vectorizerList[k].clone()
            self.vectorizerList[k].fit(X, y)
            if self.classifierList[k].useEmbedding is True:
                self.classifierList[k].numUniqueWord = len(self.vectorizerList[k].tokenizer.word_index)
                self.classifierList[k].embdeddingMatrix = self.vectorizerList[k].embeddingMatrix
            X_transformed = self.vectorizerList[k].transform(X)

            # Train classifier
            self.classifierList[k] = self.classifierList[k].clone()
            epoch_num = None
            if len(self.classifierList[k].history) > 0:
                max_epoch_nums = [len(h.epoch) for h in self.classifierList[k].history]
                epoch_num = int(np.mean(max_epoch_nums))
            if self.classifierList[k].isKerasClf is True and epoch_num is None:
                epoch_num = 10
            self.classifierList[k].fit(X=X_transformed, y=y, epochs=epoch_num)

    def predict(self, X):
        X = self.additional_preprocess(X)
        pred_prob = np.zeros((len(X),))
        numClfs = len(self.classifierList)
        for k in range(numClfs):
            # Evaluate classifier
            X_transformed = self.vectorizerList[k].transform(X)
            pred_prob += self.classifierList[k].predict_proba(X=X_transformed).squeeze()
        pred_prob /= numClfs
        return pred_prob

    def remove_overfitwords(text):
        overfit_list = ["donald trump", "hillary clinton", \
                        "donald j trump", "not", "no"]
        for ow in overfit_list:
            text = re.sub(ow, '', text)
        return text

    def additional_preprocess(self,X):
        # Earlier text normalization is not good enough
        texts = []
        for doc in X:
            special_char_pattern = re.compile(r'([^a-zA-z0-9\s]|\_)')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = re.sub('[^a-zA-z0-9\s]|\_', "", doc)
            doc = re.sub(' +', ' ', doc)
            texts.append(doc)
        return np.array(texts)

    def cross_validate(self, X, y, k, saved_folder=None):
        # X is raw texts, y is labels before encoding, k is number of folds in CV
        # useEarlyStop is used in CNN training
        label_encoder = preprocessing.LabelBinarizer()
        y = label_encoder.fit_transform(y)
        X = self.additional_preprocess(X)
        train_val = ["train","val"]
        score_names = ["acc","auc","f1","precision","recall"]
        scores = {}
        for f in train_val:
            for s in score_names:
                scores[f+"_"+s] = []
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        i = 0
        for train_index, val_index in skf.split(X, y):
            print("CV round %d..." % i)
            i += 1
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            numClfs = len(self.classifierList)
            for k in range(numClfs):
                # Train vectorizerList
                self.vectorizerList[k] = self.vectorizerList[k].clone()
                self.vectorizerList[k].fit(X_train, y_train)
                if self.classifierList[k].useEmbedding is True:
                    self.classifierList[k].numUniqueWord = len(self.vectorizerList[k].tokenizer.word_index)
                    self.classifierList[k].embdeddingMatrix = self.vectorizerList[k].embeddingMatrix
                X_train_transformed, X_val_transformed = self.vectorizerList[k].transform(X_train), self.vectorizerList[k].transform(X_val)

                # Train classifier
                self.classifierList[k] = self.classifierList[k].clone()
                if self.classifierList[k].useValEarlyStop is True:
                    history = self.classifierList[k].fit(X=X_train_transformed, y=y_train, validation_data=(X_val_transformed,y_val))
                else:
                    history = self.classifierList[k].fit(X=X_train_transformed, y=y_train)
                if history is not None:
                    self.classifierList[k].history.append(history)

            val_pred_prob = np.zeros((len(y_val),))
            train_pred_prob = np.zeros((len(y_train),))
            for k in range(numClfs):
                # Evaluate classifier
                X_train_transformed, X_val_transformed = \
                    self.vectorizerList[k].transform(X_train), self.vectorizerList[k].transform(X_val)
                val_pred_prob += self.classifierList[k].predict_proba(X=X_val_transformed).squeeze()
                train_pred_prob += self.classifierList[k].predict_proba(X=X_train_transformed).squeeze()
            val_pred_prob /= numClfs
            train_pred_prob /= numClfs
            val_pred = val_pred_prob > 0.5
            train_pred = train_pred_prob > 0.5

            for f in train_val:
                if f == "train":
                    y_true, y_eval, y_eval_prob = y_train, train_pred, train_pred_prob
                else:
                    y_true, y_eval, y_eval_prob = y_val, val_pred, val_pred_prob
                for s in score_names:
                    if s == "acc":
                        scores[f + "_" + s].append(metrics.accuracy_score(y_true, y_eval))
                    if s == "auc":
                        scores[f + "_" + s].append(metrics.roc_auc_score(y_true, y_eval_prob))
                    if s == "f1":
                        scores[f + "_" + s].append(metrics.f1_score(y_true, y_eval))
                    if s == "precision":
                        scores[f + "_" + s].append(metrics.precision_score(y_true, y_eval))
                    if s == "recall":
                        scores[f + "_" + s].append(metrics.recall_score(y_true, y_eval))
        # Summmary
        df_scores = pd.DataFrame(scores)
        df_scores.index.name = "CV round"
        df_scores = df_scores.T
        df_scores["mean"] = df_scores.mean(axis=1)
        df_scores["std"] = df_scores.std(axis=1)
        if saved_folder is None:
            return df_scores
        if os.path.isdir(saved_folder) is False:
            os.mkdir(saved_folder)
        with open(saved_folder+"/cv_score.pickle", 'wb') as handle:
            pickle.dump(df_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(saved_folder+"/cv_score.txt", 'w') as handle:
            handle.write(saved_folder+"\n")
            handle.write(str(df_scores))
            handle.write("\n")
        return df_scores

    def save_models(self, folder_path):
        if os.path.isdir(folder_path) is True:
            files = glob.glob(folder_path+"/*")
            for f in files:
                file_name = f.split(sep="/")[-1]
                if "model" in file_name or "vec" in file_name:
                    os.remove(f)
            pass
        else:
            os.mkdir(folder_path)
        numClfs = len(self.classifierList)
        print("saving models and vectorizer to " + folder_path)
        for k in range(numClfs):
            self.classifierList[k].save(folder_path+"/"+str(k)+".model")
            self.vectorizerList[k].save(folder_path+"/"+str(k)+".vec")

    def load_models(self, folder_path):
        print("loading models and vectorizer from " + folder_path)
        if os.path.isdir(folder_path) is True:
            files = sorted(glob.glob(folder_path+"/*"))
            for f in files:
                file_name = f.split(sep="/")[-1]
                if "model" in file_name:
                    index = int(file_name.split(sep=".")[0])
                    self.classifierList[index].load(f)
                if "vec" in file_name:
                    index = int(file_name.split(sep=".")[0])
                    self.vectorizerList[index].load(f)
        else:
            print(folder_path + " is empty!")
            raise ValueError