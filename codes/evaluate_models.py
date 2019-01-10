from ClassifierWrapper import *
from Vectorizer import *
from sklearn import metrics
from TextClassifier import TextClassifier
import pandas as pd
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
# warnings.filterwarnings('ignore')

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

model_name = "cntvec_lr"
saved_folder = "../saved_models/" + model_name
vec = VectorizerCountVec()
clf = LogisticRegressionWrapper(C=4, dual=True)
tc = TextClassifier(vectorizerList=[vec], classifierList=[clf])
scores = tc.cross_validate(X,y,fold_num,saved_folder=saved_folder)
tc.fit(X,y)
tc.save_models(saved_folder)

model_name = "cntvecnb_lr"
saved_folder = "../saved_models/" + model_name
vec = VectorizerCountVecNB()
clf = LogisticRegressionWrapper(C=4, dual=True)
tc = TextClassifier(vectorizerList=[vec], classifierList=[clf])
scores = tc.cross_validate(X,y,fold_num,saved_folder=saved_folder)
tc.fit(X,y)
tc.save_models(saved_folder)
"""
how to load saved models
tc2 = TextClassifier(vectorizerList=[vec], classifierList=[clf])
tc2.load_models("../saved_models/cntvecnb_lr")
pred = tc2.predict(X)
print(metrics.f1_score(y, pred>0.5))
"""

model_name = "tfidf_lr"
saved_folder = "../saved_models/" + model_name
vec = VectorizerTFIDF()
clf = LogisticRegressionWrapper(C=4, dual=True)
tc = TextClassifier(vectorizerList=[vec], classifierList=[clf])
scores = tc.cross_validate(X,y,fold_num,saved_folder=saved_folder)
tc.fit(X,y)
tc.save_models(saved_folder)

model_name = "tfidfnb_lr"
saved_folder = "../saved_models/" + model_name
vec = VectorizerTFIDFNB()
clf = LogisticRegressionWrapper(C=4, dual=True)
tc = TextClassifier(vectorizerList=[vec], classifierList=[clf])
scores = tc.cross_validate(X,y,fold_num,saved_folder=saved_folder)
tc.fit(X,y)
tc.save_models(saved_folder)

model_name = "cntvec_mnb"
saved_folder = "../saved_models/" + model_name
vec = VectorizerCountVec()
clf = MultinomialNBWrapper()
tc = TextClassifier(vectorizerList=[vec], classifierList=[clf])
scores = tc.cross_validate(X,y,fold_num,saved_folder=saved_folder)
tc.fit(X,y)
tc.save_models(saved_folder)

model_name = "tfidfnb_rf"
saved_folder = "../saved_models/" + model_name
vec = VectorizerTFIDFNB()
clf = RandomForestClassifierWrapper(n_estimators=50,
                                    max_features=0.8,
                                    random_state=42,n_jobs=-1)
tc = TextClassifier(vectorizerList=[vec], classifierList=[clf])
scores = tc.cross_validate(X,y,fold_num,saved_folder=saved_folder)
tc.fit(X,y)
tc.save_models(saved_folder)

model_name = "tfidfnb_svmlinear"
saved_folder = "../saved_models/" + model_name
vec = VectorizerTFIDFNB()
clf = SVCWrapper(kernel='linear',probability=True)
tc = TextClassifier(vectorizerList=[vec], classifierList=[clf])
scores = tc.cross_validate(X,y,fold_num,saved_folder=saved_folder)
tc.fit(X,y)
tc.save_models(saved_folder)

model_name = "tfidfnb_lrbagging"
saved_folder = "../saved_models/" + model_name
vec = VectorizerTFIDFNB()
clf = BaggingClassifierWrapper(base_estimator=LogisticRegression(),
                               n_estimators=50,
                               bootstrap=True,
                               bootstrap_features=True,
                               verbose=1,
                               n_jobs=-1)
tc = TextClassifier(vectorizerList=[vec], classifierList=[clf])
scores = tc.cross_validate(X,y,fold_num,saved_folder=saved_folder)
tc.fit(X,y)
tc.save_models(saved_folder)

model_name = "ensembled_tfidfnb_lr_cntvec_mnb"
saved_folder = "../saved_models/" + model_name
tc = TextClassifier(vectorizerList=[VectorizerTFIDFNB(), VectorizerCountVec()],
                    classifierList=[LogisticRegressionWrapper(C=4, dual=True), MultinomialNBWrapper()])
scores = tc.cross_validate(X,y,fold_num,saved_folder=saved_folder)
tc.fit(X,y)
tc.save_models(saved_folder)
