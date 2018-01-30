import argparse
from embeddings import read_embeddings, cutten_embeddings
from text_processing import tokenize, text_to_sequence
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.layers import InputLayer, Embedding, CuDNNGRU, Bidirectional, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.metrics import log_loss
from utils import get_wordset
import os
import gc


MAXLEN = 500
FOLD_COUNT = 10


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("submission")
    parser.add_argument("embedding")
    parser.add_argument("modeldir")
    parser.add_argument("fold")
    parser.add_argument("test_prediction")
    return parser.parse_args()



def fit_models(X, y, embedding_vectors, fname, fold):

    def _get_embedding():
        return Embedding(embedding_vectors.shape[0],
                         embedding_vectors.shape[1],
                         weights=[embedding_vectors],
                         trainable=False)

    def _get_model():
        model = Sequential([
            InputLayer(input_shape=(MAXLEN,), dtype='int32'),
            _get_embedding(),
            Bidirectional(CuDNNGRU(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(CuDNNGRU(64, return_sequences=False)),
            Dense(32, activation='relu'),
            Dense(y.shape[1], activation='sigmoid')
        ])
        model.compile(optimizer=RMSprop(clipvalue=1, clipnorm=1),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def _mean_log_loss(y_true, y_pred):
        labels = y_true.shape[1]
        errors = np.zeros([labels])
        for i in range(labels):
            errors[i] = log_loss(y_true[:, i], y_pred[:, i])
        return errors.mean()

    def _fold_fit(X_train, X_val, y_train, y_val, model):
        best_loss = -1
        best_weights = None
        best_epoch = 0
        current_epoch = 0

        while True:
            print("Epoch {0}".format(current_epoch + 1))
            model.fit(X_train, y_train, batch_size=256, epochs=1, verbose=True)
            loss = _mean_log_loss(y_val, model.predict(X_val, batch_size=256))
            print("Loss: {0}".format(loss))
            current_epoch += 1
            if loss < best_loss or best_loss == -1:
                best_loss = loss
                best_weights = model.get_weights()
                best_epoch = current_epoch
            else:
                if current_epoch - best_epoch == 5:
                    break
        print("Best loss {0} on epoch {1}".format(best_loss, best_epoch))
        model.set_weights(best_weights)

        return model

    fold_size = len(X) // FOLD_COUNT
    model = _get_model()
    initial_weights = model.get_weights()

    print("Fold {0}".format(fold + 1))
    start = fold_size * fold
    end = start + fold_size
    if fold == FOLD_COUNT - 1:
        end = len(X)

    X_train = np.concatenate([X[:start], X[end:]])
    X_val = X[start:end]
    y_train = np.concatenate([y[:start], y[end:]])
    y_val = y[start:end]

    model.set_weights(initial_weights)
    model = _fold_fit(X_train, X_val, y_train, y_val, model)
    model.save(fname)


if __name__ == '__main__':
    args = get_args()
    targets = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    print("Reading embeddings")
    word2index, vectors = read_embeddings(args.embedding)

    print("Reading texts")
    dftrain = pd.read_csv(args.train)
    dftest = pd.read_csv(args.test)
    texts = list(dftrain['comment_text']) + list(dftest['comment_text'])

    print("Reading submission")
    submission = pd.read_csv(args.submission)

    print("Getting words")
    wordset = get_wordset(texts)

    print("Cutting embeddings")
    word2index_cut, vectors_cut = cutten_embeddings(wordset, word2index, vectors)
    del word2index, vectors
    gc.collect()

    print("Converting texts")
    X_train = np.array([
        text_to_sequence(text, word2index_cut, MAXLEN)
        for text in tqdm(dftrain['comment_text'])
    ])
    X_test = np.array([
        text_to_sequence(text, word2index_cut, MAXLEN)
        for text in tqdm(dftest['comment_text'])
    ])

    print("Converting outputs")
    y_train = np.array(dftrain[targets])

    print("Fitting models")
    os.makedirs(args.modeldir, exist_ok=True)
    fnames = [os.path.join(args.modeldir, str(i) + ".h5") for i in range(FOLD_COUNT)]
    fold_index = int(args.fold) - 1
    fit_models(X_train, y_train, vectors_cut, fnames[fold_index], fold_index)
