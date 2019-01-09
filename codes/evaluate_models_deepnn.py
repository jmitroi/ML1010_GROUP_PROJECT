from ClassifierWrapper import *
from Vectorizer import *
from sklearn import metrics
from sklearn.model_selection import train_test_split
from TextClassifier import TextClassifier
import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# Macro control
fold_num = 5 # kfold
downsamlping = False

# Read data
np.random.seed(0)
df = pd.read_csv("../data/normalized_texts_labels.csv",encoding="utf-8")
df = df[["normalized_title","normalized_text","fake"]]
df.columns = ["titles","texts","labels"]
print("# of NaN of texts:" + str(df["texts"].isnull().sum()))
print("# of NaN of labels:" + str(df["labels"].isnull().sum()))
print("# of NaN of titles:" + str(df["titles"].isnull().sum()))
df = df.dropna()

# downsampling
if downsamlping is True:
    df = df.iloc[list(range(0,df.shape[0],100))]
print("dataset size:" + str(df.shape))
y = df["labels"].values
X = df["texts"].values

# hold out one set as final test set
X, X_test, y, y_test = train_test_split(X, y, stratify=y, random_state=12345, test_size=0.2, shuffle=True)

model_name = "glove_cnn"
saved_folder = "../saved_models/" + model_name
vec = VectorizerEmbedding(docLen=5000,word_vector_file="../wordvecs/glove.6B.50d.txt")
clf = CNNWrapper(docLen=5000)
tc = TextClassifier(vectorizerList=[vec], classifierList=[clf])
scores = tc.cross_validate(X,y,fold_num,saved_folder=saved_folder)
print("Fitting classifier(s) on the whole dataset.")
tc.fit(X,y)
tc.save_models(saved_folder)
pred_test = tc.predict(X_test)
print(model_name + "accuracy on testset")
print(metrics.f1_score(y_test, pred_test>0.5))

"""
how to load saved models
tc2 = TextClassifier(vectorizerList=[vec], classifierList=[clf])
tc2.load_models(saved_folder)
pred = tc2.predict(X)
print(metrics.f1_score(y, pred>0.5))
"""

model_name = "fasttext_cnn"
saved_folder = "../saved_models/" + model_name
vec = VectorizerEmbedding(docLen=5000,word_vector_file="../wordvecs/glove.6B.50d.txt")
clf = CNNWrapper(docLen=5000)
tc = TextClassifier(vectorizerList=[vec], classifierList=[clf])
scores = tc.cross_validate(X,y,fold_num,saved_folder=saved_folder)
print("Fitting classifier(s) on the whole dataset.")
tc.fit(X,y)
tc.save_models(saved_folder)
pred_test = tc.predict(X_test)
print(model_name + "accuracy on testset")
print(metrics.f1_score(y_test, pred_test>0.5))

model_name = "ensemble_fasttext_cnn_tfidfnb_lr"
saved_folder = "../saved_models/" + model_name
tc = TextClassifier(vectorizerList=[VectorizerTFIDFNB(), VectorizerCountVec()],
                    classifierList=[LogisticRegressionWrapper(C=4, dual=True), LogisticRegressionWrapper(C=4, dual=True)])
scores = tc.cross_validate(X,y,fold_num,saved_folder=saved_folder)
print("Fitting classifier(s) on the whole dataset.")
tc.fit(X,y)
tc.save_models(saved_folder)
pred_test = tc.predict(X_test)
print(model_name + "accuracy on testset")
print(metrics.f1_score(y_test, pred_test>0.5))
