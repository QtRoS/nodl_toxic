from argparse import ArgumentParser
from embeddings import read_embeddings, cutten_embeddings
import gc
import os
from glob import glob
from utils import get_wordset
from text_processing import text_to_sequence
import pandas as pd
from tqdm import tqdm
import numpy as np
from keras.models import load_model
from keras import backend as K

from model_train import custom_loss

MAXLEN = 500


def get_args():
    parser = ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("embedding")
    parser.add_argument("model", help="model path or glob")
    parser.add_argument("output", help="output filename or directory (if model is glob)")
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

    def predict_single_model(model_path, output_path):
        print("Making prediction for model:", model_path)
        model = load_model(model_path, custom_objects={'custom_loss': custom_loss})
        prediction = model.predict(X_test, batch_size=256, verbose=True)
        print("Saving prediction:", output_path)
        np.save(output_path, prediction)
        del model, prediction
        K.clear_session()
        gc.collect()
        

    if not os.path.exists(args.model):
        print("Model not found, assuming that it is glob:", args.model)
        for mdl in glob(args.model):  # TODO sorted?
            predict_single_model(mdl, os.path.join(args.output, os.path.basename(mdl) + '.npy'))
    else:
        predict_single_model(args.model, args.output)
