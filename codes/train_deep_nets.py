# Modeling related
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
import pandas as pd, numpy
from keras.preprocessing import text, sequence
from model_zoos import CNN, LSTM
import pickle

def creat_vector_features(word_vector_file, texts):
    """
    :param word_vector_file: path to word2vec file
    :param texts: a list of normailzed documents
    :return:
    """
    embeddings_index = {}
    for i, line in enumerate(open(word_vector_file)):
        values = line.split()
        embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')
    token = text.Tokenizer()
    token.fit_on_texts(texts)
    word_index = token.word_index

    # convert text to sequence of tokens and pad them to ensure equal length vectors
    text_tokens = token.texts_to_sequences(texts)
    max_tokens_one_sent = 0
    min_tokens_one_sent = 1000000000
    for doc in text_tokens:
        max_tokens_one_sent = max(max_tokens_one_sent, len(doc))
        min_tokens_one_sent = min(min_tokens_one_sent, len(doc))
    print("Max # of tokens in docs: " + str(max_tokens_one_sent))
    print("Min # of tokens in docs: " + str(min_tokens_one_sent))
    text_seqences = sequence.pad_sequences(text_tokens, maxlen=max_tokens_one_sent)
    # create token-embedding mapping
    embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return text_seqences,  embedding_matrix, word_index, max_tokens_one_sent

def main():
    # read normailzed texts & labels, subsample to run on local machines
    df = pd.read_csv("../data/normalized_texts_labels.csv")
    df = df[["normalized_text", "fake"]]
    df.columns = ["texts", "labels"]
    #df = df.iloc[list(range(0,df.shape[0],10))]
    print("# of NaN of text:" + str(df["texts"].isnull().sum()))
    print("# of NaN of label:" + str(df["labels"].isnull().sum()))
    df = df.dropna()
    print("dataset size:" + str(df.shape))

    # Encode labels
    encoder = preprocessing.LabelBinarizer()
    encoder.fit(df["labels"])
    encoded_y = encoder.transform(df["labels"])

    # Convert texts to vector representation
    text_seqences, embedding_matrix, word_index, max_tokens_one_sent = \
        creat_vector_features('../wordvecs/wiki-news-300d-1M.vec', df["texts"])

    # train validation split
    train_seq_x, valid_seq_x, train_encoded_y, valid_encoded_y = \
        model_selection.train_test_split(text_seqences, encoded_y, stratify=encoded_y)

    # CNN model
    cnn = CNN(len(word_index)+1, embedding_matrix,
              max_tokens_one_sent, len(encoder.classes_))
    cnn_model = cnn.create_model()
    history = cnn_model.fit(x=train_seq_x, y=train_encoded_y, epochs=10)
    predictions = cnn_model.predict(valid_seq_x)
    print("CNN accuracy on validation set:")
    print(metrics.accuracy_score(predictions.argmax(axis=1), valid_encoded_y.argmax(axis=1)))
    model_file_names = "../saved_models/cnn.model"
    print("saving models to " + model_file_names)
    cnn_model.save(model_file_names)
    with open('../saved_models/cnn.model.history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # LSTM model
    lstm = LSTM(len(word_index) + 1, embedding_matrix,
              max_tokens_one_sent, len(encoder.classes_))
    lstm_model = lstm.create_model()
    history = lstm_model.fit(x=train_seq_x, y=train_encoded_y, epochs=10)
    predictions = lstm_model.predict(valid_seq_x)
    print("CNN accuracy on validation set:")
    print(metrics.accuracy_score(predictions.argmax(axis=1), valid_encoded_y.argmax(axis=1)))
    model_file_names = "../saved_models/lstm.model"
    print("saving models to " + model_file_names)
    lstm_model.save(model_file_names)
    with open('../saved_models/lstm.model.history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

if __name__ == "__main__":
    main()
