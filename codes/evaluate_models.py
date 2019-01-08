import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from TextClassifier import TextClassifier
import warnings
warnings.filterwarnings('ignore')
np.random.seed(0)
df = pd.read_csv("../data/normalized_texts_labels.csv",encoding="utf-8")
df = df[["normalized_title","normalized_text","fake"]]
df.columns = ["titles","texts","labels"]
print("# of NaN of texts:" + str(df["texts"].isnull().sum()))
print("# of NaN of labels:" + str(df["labels"].isnull().sum()))
print("# of NaN of titles:" + str(df["titles"].isnull().sum()))
df = df.dropna()
# downsampling
df = df.iloc[list(range(0,df.shape[0],80))]
print("dataset size:" + str(df.shape))
y = df["labels"].values
X = df["texts"].values

from ClassifierWrapper import *
from Vectorizer import *

"""
vec = VectorizerTFIDF(max_features=None, ngram_range=(1, 2))
clf = LogisticRegressionTemplate(C=4, dual=True)
tc = TextClassifier(vectorizer=vec, classifierTemplate=clf)
scores = tc.cross_validate(X,y,3)
"""

"""
vec = VectorizerFasttext()
clf = CNNTemplate(docLen=5000)
tc = TextClassifier(vectorizer=vec, classifierTemplate=clf)
scores = tc.cross_validate(X,y,3,useEarlyStop=True,useEmbedding=True)
"""

#vec = VectorizerCountVecNB()
#clf = LogisticRegressionWrapper(C=4, dual=True)
#tc = TextClassifier(vectorizerList=[vec], classifierList=[clf])
#scores = tc.cross_validate(X,y,3)

# , VectorizerFasttext(docLen=5000)
# , CNNWrapper(docLen=5000)
vec = VectorizerFasttext(docLen=5000)
clf = CNNWrapper(docLen=5000)
tc = TextClassifier(vectorizerList=[VectorizerTFIDFNB(), VectorizerFasttext(docLen=5000)],
                    classifierList=[LogisticRegressionWrapper(C=4, dual=True), CNNWrapper(docLen=5000)])
scores = tc.cross_validate(X,y,3)

print(scores)