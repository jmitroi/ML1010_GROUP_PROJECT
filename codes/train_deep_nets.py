# Modeling related
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
import pandas as pd, numpy as np
from keras.preprocessing import text, sequence
from keras.models import load_model
import pickle
import tensorflow as tf
from model_zoos import CNN, LSTM

def create_embedding_matrix(word_vector_file, tokenizer):
    embeddings_index = {}
    for i, line in enumerate(open(word_vector_file)):
        values = line.split()
        embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def generate_word_sequence(texts,feature_dimension,tokenizer):
    text_tokens = tokenizer.texts_to_sequences(texts)
    text_seqences = sequence.pad_sequences(text_tokens, maxlen=feature_dimension)
    return text_seqences

def main():
    # read normailzed texts & labels, subsample to run on local machines
    df = pd.read_csv("../data/normalized_texts_labels.csv")
    df = df[["normalized_text", "fake"]]
    df.columns = ["texts", "labels"]

    # downsampling
    # df = df.iloc[list(range(0,df.shape[0],80))]

    print("# of NaN of text:" + str(df["texts"].isnull().sum()))
    print("# of NaN of label:" + str(df["labels"].isnull().sum()))
    df = df.dropna()
    print("dataset size:" + str(df.shape))

    train_texts, test_texts, train_labels, test_labels = model_selection.train_test_split(df.texts, df.labels,
                                                                          stratify=df.labels,
                                                                          random_state=12345,
                                                                          test_size=0.1, shuffle=True)
    train_val_texts, train_val_labels = train_texts, train_labels
    train_texts, val_texts, train_labels, val_labels = model_selection.train_test_split(train_texts, train_labels,
                                                                        stratify=train_labels,
                                                                        random_state=12345,
                                                                        test_size=0.1, shuffle=True)

    # Encode labels
    label_encoder = preprocessing.LabelBinarizer()
    label_encoder.fit(df["labels"])
    train_labels_encoded, val_labels_encoded, test_labels_encoded, train_val_labels_encoded = \
        label_encoder.transform(train_labels), \
        label_encoder.transform(val_labels), \
        label_encoder.transform(test_labels), \
        label_encoder.transform(train_val_labels)

    # Convert texts to vector representation
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(df["texts"])
    # max_tokens_one_sent in wordvec and cnn
    text_tokens = tokenizer.texts_to_sequences(df["texts"])
    max_tokens_one_sent = 0
    min_tokens_one_sent = 1000000000
    for doc in text_tokens:
        max_tokens_one_sent = max(max_tokens_one_sent, len(doc))
        min_tokens_one_sent = min(min_tokens_one_sent, len(doc))
    print("Max # of tokens in docs: " + str(max_tokens_one_sent))
    print("Min # of tokens in docs: " + str(min_tokens_one_sent))

    # embedding matrix used in first layer of cnn
    embedding_matrix = create_embedding_matrix('../wordvecs/wiki-news-300d-1M.vec', tokenizer)
    # train validation split
    train_seq_x, valid_seq_x, test_seq_x, train_val_seq_x = \
        generate_word_sequence(train_texts, max_tokens_one_sent, tokenizer), \
        generate_word_sequence(val_texts, max_tokens_one_sent, tokenizer), \
        generate_word_sequence(test_texts, max_tokens_one_sent, tokenizer), \
        generate_word_sequence(train_val_texts, max_tokens_one_sent, tokenizer)


    # CNN model
    success = False
    print("train on training set.")
    while success == False:
        try:
            cnn = CNN(len(tokenizer.word_index) + 1, embedding_matrix,
                      max_tokens_one_sent)
            cnn_model = cnn.create_model()
            history = cnn_model.fit(x=train_seq_x, y=train_labels_encoded, epochs=10)
            predictions = cnn_model.predict(valid_seq_x)
            success = True
        except tf.errors.ResourceExhaustedError as e:
            success = False
            print("Fail to acquire resources! Retrying.")
    print("CNN accuracy on validation set:")
    print(metrics.accuracy_score(predictions > 0.5, val_labels_encoded))
    model_file_names = "../saved_models/cnn_trained_on_trainset.model"
    print("saving models to " + model_file_names)
    cnn_model.save(model_file_names)
    with open('../saved_models/cnn.model.history.trained_on_trainset', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    success = False
    print("train on training set and val set.")
    while success == False:
        try:
            cnn = CNN(len(tokenizer.word_index) + 1, embedding_matrix,
                      max_tokens_one_sent)
            cnn_model = cnn.create_model()
            history = cnn_model.fit(x=train_val_seq_x,y=train_val_labels_encoded,epochs=10)
            predictions = cnn_model.predict(test_seq_x)
            success = True
        except tf.errors.ResourceExhaustedError as e:
            success = False
            print("Fail to acquire resources! Retrying.")
    print("CNN accuracy on test set:")
    print(metrics.accuracy_score(predictions > 0.5, test_labels_encoded))
    model_file_names = "../saved_models/cnn_trained_on_trainset_and_valset.model"
    print("saving models to " + model_file_names)
    cnn_model.save(model_file_names)
    with open('../saved_models/cnn.model.history.trained_on_trainset_and_valset', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # Because LSTM takes too long, we are not training it for mid project submission
    """
    # LSTM model
    success = False
    while success == False:
        try:
            lstm = LSTM(len(word_index) + 1, embedding_matrix,
                        max_tokens_one_sent)
            lstm_model = lstm.create_model()
            history = lstm_model.fit(x=train_seq_x, y=train_encoded_y, epochs=10)
            predictions = lstm_model.predict(valid_seq_x)
            success = True
        except tf.errors.ResourceExhaustedError as e:
            success = False
            print("Fail to acquire resources! Retrying.")
    print("LSTM accuracy on validation set:")
    print(metrics.accuracy_score(predictions > 0.5, valid_encoded_y))
    model_file_names = "../saved_models/lstm.model"
    print("saving models to " + model_file_names)
    lstm_model.save(model_file_names)
    with open('../saved_models/lstm.model.history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    """


if __name__ == "__main__":
    main()
