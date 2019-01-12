from ClassifierWrapper import *
from Vectorizer import *
from sklearn import metrics
from TextClassifier import TextClassifier
import pandas as pd
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
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

model_set = {
    "lr_cntvec": {
        "vec": [VectorizerCountVec()],
        "clf": [LogisticRegressionWrapper(C=4, dual=True)],
        "run": False
    },
    "lr_cntvecnb": {
        "vec": [VectorizerCountVecNB()],
        "clf": [LogisticRegressionWrapper(C=4, dual=True)],
        "run": False
    },
    "lr_tfidf": {
        "vec": [VectorizerTFIDF()],
        "clf": [LogisticRegressionWrapper(C=4, dual=True)],
        "run": False
    },
    "lr_tfidfnb": {
        "vec": [VectorizerTFIDFNB()],
        "clf": [LogisticRegressionWrapper(C=4, dual=True)],
        "run": False
    },
    "mnb_cntvec": {
        "vec": [VectorizerCountVec()],
        "clf": [MultinomialNBWrapper()],
        "run": False
    },
    "rf_cntvec": {
        "vec": [VectorizerCountVec()],
        "clf": [RandomForestClassifierWrapper(n_estimators=50,
                                              max_features=0.8,
                                              random_state=42,n_jobs=-1)],
        "run": False
    },
    "rf_cntvecnb": {
        "vec": [VectorizerCountVecNB()],
        "clf": [RandomForestClassifierWrapper(n_estimators=50,
                                              max_features=0.8,
                                              random_state=42,n_jobs=-1)],
        "run": False
    },
    "rf_tfidf": {
        "vec": [VectorizerTFIDF()],
        "clf": [RandomForestClassifierWrapper(n_estimators=50,
                                              max_features=0.8,
                                              random_state=42,n_jobs=-1)],
        "run": False
    },
    "rf_tfidfnb": {
        "vec": [VectorizerTFIDFNB()],
        "clf": [RandomForestClassifierWrapper(n_estimators=50,
                                              max_features=0.8,
                                              random_state=42,n_jobs=-1)],
        "run": False
    },
    "svmlinear_tfidf": {
        "vec": [VectorizerTFIDF()],
        "clf": [SVCWrapper(kernel='linear',probability=True)],
        "run": False
    },
    "svmlinear_tfidfnb": {
        "vec": [VectorizerTFIDFNB()],
        "clf": [SVCWrapper(kernel='linear',probability=True)],
        "run": False
    },
    "svmlinear_cntvec": {
        "vec": [VectorizerCountVec()],
        "clf": [SVCWrapper(kernel='linear',probability=True)],
        "run": True
    },
    "svmlinear_cntvecnb": {
        "vec": [VectorizerCountVecNB()],
        "clf": [SVCWrapper(kernel='linear',probability=True)],
        "run": True
    },
    "lrbagging_tfidf": {
        "vec": [VectorizerTFIDF()],
        "clf": [BaggingClassifierWrapper(base_estimator=LogisticRegression(),
                                         n_estimators=50,
                                         bootstrap=True,
                                         bootstrap_features=True,
                                         verbose=1,
                                         n_jobs=-1)],
        "run": False
    },
    "lrbagging_tfidfnb": {
        "vec": [VectorizerTFIDFNB()],
        "clf": [BaggingClassifierWrapper(base_estimator=LogisticRegression(),
                                         n_estimators=50,
                                         bootstrap=True,
                                         bootstrap_features=True,
                                         verbose=1,
                                         n_jobs=-1)],
        "run": False
    },
    "lrbagging_cntvec": {
        "vec": [VectorizerCountVec()],
        "clf": [BaggingClassifierWrapper(base_estimator=LogisticRegression(),
                                         n_estimators=50,
                                         bootstrap=True,
                                         bootstrap_features=True,
                                         verbose=1,
                                         n_jobs=-1)],
        "run": False
    },
    "lrbagging_cntvecnb": {
        "vec": [VectorizerCountVecNB()],
        "clf": [BaggingClassifierWrapper(base_estimator=LogisticRegression(),
                                         n_estimators=50,
                                         bootstrap=True,
                                         bootstrap_features=True,
                                         verbose=1,
                                         n_jobs=-1)],
        "run": False
    }
}

num_models_to_train = 0
for k, v in model_set.items():
    if v["run"] is False:
        continue
    num_models_to_train += 1
print("Total models to train: " + str(num_models_to_train))

for k, v in model_set.items():
    if v["run"] is False:
        continue
    model_name = k
    saved_folder = "../saved_models/" + model_name
    vec_list = v["vec"]
    clf_list = v["clf"]
    tc = TextClassifier(vectorizerList=vec_list, classifierList=clf_list)
    scores = tc.cross_validate(X, y, fold_num, saved_folder=saved_folder)
    tc.fit(X, y)
    tc.save_models(saved_folder)

print("Done!")

"""
how to load saved models
tc2 = TextClassifier(vectorizerList=[vec], classifierList=[clf])
tc2.load_models("../saved_models/cntvecnb_lr")
pred = tc2.predict(X)
print(metrics.f1_score(y, pred>0.5))
"""

"""
model_name = "ensembled_tfidfnb_lr_cntvec_mnb"
saved_folder = "../saved_models/" + model_name
tc = TextClassifier(vectorizerList=[VectorizerTFIDFNB(), VectorizerCountVec()],
                    classifierList=[LogisticRegressionWrapper(C=4, dual=True), MultinomialNBWrapper()])
scores = tc.cross_validate(X,y,fold_num,saved_folder=saved_folder)
tc.fit(X,y)
tc.save_models(saved_folder)
"""
