# ML1010_GROUP_PROJECT
Fake news detection

Relevant files Mid-project proposal:
write_up.pdf
The write-up for the proposal. It's the primary place we document our steps taken for the task.

codes/normailization.py
Codes to normalize the news body and news title.

codes/train_deep_nets.py
Codes to train CNN.

notebooks/News_DataPrep_EDA.ipynb
Codes that are used to generate dataset for fake news and real news.

notebooks/model_evaluations.ipynb
Codes for feature extractions/engineering, model tranining, and model evaluation.

notebooks/Feature_Selection.ipynb
Codes for feature selections. We have spent some effort into this; but note that we have not integrated it into the machine learning pipeline yet.

Notes:
1. Github does not allow files that have size >100MB to be uploaded. Therefor, to repeat the results and run codes/notbooks, you need to uncompress the files stored in these two folders:
saved_models/
Folder that stores trained models(CNN)
data/
Folder that stores data
2. To repeat the results and run code, you also need to download word vectors from this address:
https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip
Then uncompress the file and put into the folder of "wordvecs"
3. Run training of deep nets (CNN) as follows:
python3 codes/train_deep_nets.py
