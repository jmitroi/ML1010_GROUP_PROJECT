# ML1010_GROUP_PROJECT
Project name: fake news detection   
Final report is located at write_up/final_report.pdf   


Here are a list of descriptions for each folder in this reposition.   
1. write_up/   

Project proposal write-up and final write-up, as well as powerpoint slides used for in-class presentations.

2. codes/   

ClassifierWrapper.py   
Wrapper for all classifiers used in the project.     

Vectorizer.py   
Wrapper for all vectorizer used in the project.

TextClassifier.py   
A class that is initialized with classifiers from ClassifierWrapper.py and vectorizers from Vectorizer.py. Classifiers and vectorizers are passed into the constructor of this class as lists. That means, more than one kind of classifiers/vectorizers can be passed into this class. In such case, the probability outputs of all classifiers are averaged and provided as the final probability prediction. This class also provide a method to do k-fold cross-validation and return a dataframe of relevant scores. The scores are stored to disk if a folder is specified.

evaluate_models.py   
This file documents the models that we have trained other than CNN.

evaluate_models_deepnn.py   
This file documents exclusively training of CNN models.

normailization.py   
This file documents the steps taken to normailze the news body. This file will normalize the text found in "real_fake_news.csv" that is generated in News_DataPrep_EDA.ipynb.

ModelingResults.ipynb   
A notebook that shows the comparison of accuracies of all the models that we have trained, as well as a simple case analysis on the texts that get classified incorrectly. Note that because the models are too big to save in github, reviewer of this project will not be able to load the models that we have trained during the project. We suggest to look directly at the results instead of trying to run this notebook.

News_DataPrep_EDA.ipynb   
A notebook that documents the steps to generate the dataset "real_fake_news.csv" used in normalization.py. It also contains some exploratory data visualization. In order to run this notebook, you will also need to download news dataset from these two sources:   
https://www.kaggle.com/snapcrack/all-the-news/home   
https://www.kaggle.com/mrisdal/fake-news   
And put them into "data" folder.   

3. saved_models/   

This folder save the cross_validation scores for all models.

4. wordvecs/    

This folder stores the pretrained word vectors downloaded from the web and used in this project. We have used the following two pretrained word vectors in our experiments:     
fastText:   
https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip   
GloVe:    
kaggle datasets download -d rdizzl3/glove6b50d   
If reviwer want to run train models using the codes from codes/, you will need to create this folder and download these pretrained vectors to you computer yourself. They are too big for github.


5. data/   

Data used in this project. Github does not allow files that have size >100MB to be uploaded. Therefor, to repeat the results and run codes/notbooks, you need to uncompress the files stored in this folder.
