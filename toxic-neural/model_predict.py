from argparse import ArgumentParser
from embeddings import read_embeddings, cutten_embeddings
import gc
from utils import get_wordset
from text_processing import text_to_sequence
import pandas as pd
from tqdm import tqdm
import numpy as np
from keras.models import load_model


MAXLEN = 500


def get_args():
    parser = ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("embedding")
    parser.add_argument("model")
    parser.add_argument("output")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    print("Reading embeddings")
    word2index, vectors = read_embeddings(args.embedding)

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
    X_test = np.array([
        text_to_sequence(text, word2index_cut, MAXLEN)
        for text in tqdm(dftest['comment_text'])
    ])

    print("Making prediction")
    model = load_model(args.model)
    prediction = model.predict(X_test, batch_size=256, verbose=True)

    print("Saving prediction")
    np.save(args.output, prediction)
