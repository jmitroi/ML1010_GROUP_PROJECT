# ML1010_GROUP_PROJECT
Project name: fake news detection

Relevant files Mid-project proposal reviewer(s):   
ML1010 - JSMCJ Group Project Proposal.pdf  
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
data/     

2. To repeat the results and run code, you also need to download word vectors from this address:   
https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip   
Also GloVe:    
kaggle datasets download -d rdizzl3/glove6b50d   

Then uncompress the file and put into the folder of "wordvecs".   
4. Run text normalization as follows:   
python3 codes/normalization.py  
It will normalize the news body and title. But for the proposal, we only use the news body as the input to the classifier.
3. Run training of deep nets (CNN) as follows:   
python3 codes/train_deep_nets.py   
4. To run notebooks/News_DataPrep_EDA.ipynb and generate the dataset "real_fake_news.csv" used in normalization.py, you will also need to download news dataset from these two sources:
https://www.kaggle.com/snapcrack/all-the-news/home
https://www.kaggle.com/mrisdal/fake-news
And put them into "data" folder.
