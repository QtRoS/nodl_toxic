from argparse import ArgumentParser
from embeddings import read_embeddings, cutten_embeddings
import pandas as pd
from text_processing import text_to_sequence
from utils import get_wordset
import gc
from tqdm import tqdm
import numpy as np
import os
from keras.models import load_model
from collections import OrderedDict


MAXLEN = 500
FOLD_COUNT = 10


def get_args():
    parser = ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("embeddings")
    parser.add_argument("modeldir")
    parser.add_argument("valdir")
    parser.add_argument("fold")
    return parser.parse_args()


def fold_boundaries(X, fold):
    fold_size = len(X) // FOLD_COUNT
    start = fold_size * fold
    end = start + fold_size
    if fold == FOLD_COUNT - 1:
        end = len(X)
    return start, end



if __name__ == '__main__':
    args = get_args()

    print("Reading embeddings")
    word2index, vectors = read_embeddings(args.embeddings)

    print("Reading texts")
    dftrain = pd.read_csv(args.train)
    dftest = pd.read_csv(args.test)
    texts = list(dftrain['comment_text']) + list(dftest['comment_text'])

    print("Getting words")
    wordset = get_wordset(texts)

    print("Cutting embeddings")
    word2index_cut, vectors_cut = cutten_embeddings(wordset, word2index, vectors)
    del word2index, vectors
    gc.collect()

    print("Converting texts")
    X = np.array([
        text_to_sequence(text, word2index_cut, MAXLEN)
        for text in tqdm(dftrain['comment_text'])
    ])

    val_start, val_end = fold_boundaries(X, int(args.fold))
    X_val = X[val_start:val_end]
    texts = np.array(dftrain['comment_text'])[val_start:val_end]
    target_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    targets = np.array(dftrain[target_columns])[val_start:val_end]

    print("Prediction")
    model = load_model(os.path.join(args.modeldir, "{0}.h5".format(args.fold)))
    prediction_val = model.predict(X_val, verbose=True) ** 1.4

    print("Saving")
    result = pd.DataFrame(OrderedDict(
        [('text', texts)] +
        [(column, targets[:, i]) for i, column in enumerate(target_columns)] +
        [(column + '_predicted', prediction_val[:, i]) for i, column in enumerate(target_columns)]
    ))
    result.to_csv(os.path.join(args.valdir, '{0}.csv'.format(args.fold)))