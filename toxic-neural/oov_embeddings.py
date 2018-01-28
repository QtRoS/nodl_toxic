import argparse
from copy import deepcopy
import pandas as pd
import numpy as np
import json
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from fuzzyset import FuzzySet
from embeddings import read_embeddings, save_embeddings
from text_processing import tokenize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("embedding")
    parser.add_argument("output")
    parser.add_argument("mapping")
    args = parser.parse_args()
    return args


def get_wordset(texts):
    return set(chain(*map(tokenize, texts)))


def get_vocabulary_word_weights(vocabulary_words, texts):
    tfidf = TfidfVectorizer(min_df=3, max_features=100000, tokenizer=tokenize)
    tfidf_values = tfidf.fit_transform(texts)
    features = set(tfidf.get_feature_names()) & vocabulary_words
    weights = {}
    feature_name_dict = {k: v for v, k in enumerate(tfidf.get_feature_names())}
    tfidf_values_col_means = np.array(tfidf_values.mean(axis=0)).ravel()
    for word in tqdm(features):
        weights[word] = tfidf_values_col_means[feature_name_dict[word]]  #tfidf_values[:, feature_name_dict[word]].mean()
    return weights


def get_oov_vocabulary_map(vocabulary_words_weights, wordset):
    oov = wordset - set(vocabulary_words_weights.keys())
    vocabulary_words_set = FuzzySet(sorted(vocabulary_words_weights.keys()))
    mapping = {}
    for word in tqdm(oov):
        word_matches = vocabulary_words_set.get(word)
        if word_matches is None or len(word_matches) == 0:
            continue
        word_scores = {vocabulary_word: score * vocabulary_words_weights[vocabulary_word]
                       for score, vocabulary_word in word_matches}
        vocabulary_words_scored = sorted(word_scores.keys(),
                                         key=lambda vocabulary_word: -word_scores[vocabulary_word])
        mapping[word] = vocabulary_words_scored[0]
    return mapping


def save_oov_mapping(oov_vocabulary_map, path):
    with open(path, 'w', encoding='utf-8') as target:
        json.dump(oov_vocabulary_map, target)


if __name__ == '__main__':
    args = parse_args()

    print("Reading embedding:")
    embeddings_word2index, embeddings_vectors = read_embeddings(args.embedding)
    vocabulary_words = set(embeddings_word2index.keys())

    print("Reading texts")
    dftrain = pd.read_csv(args.train)
    dftest = pd.read_csv(args.test)
    texts = list(dftrain['comment_text']) + list(dftest['comment_text'])

    print("Getting words")
    wordset = get_wordset(texts)

    print("Getting vocabulary word weights")
    vocabulary_words_weights = get_vocabulary_word_weights(vocabulary_words, texts)

    print("Computing OOV-vocabulary map")
    oov_vocabulary_map = get_oov_vocabulary_map(vocabulary_words_weights, wordset)

    print("Extending embedding")
    word2index = deepcopy(embeddings_word2index)
    for word, vocabulary_word in oov_vocabulary_map.items():
        word2index[word] = word2index[vocabulary_word]

    print("Saving new embedding")
    save_embeddings(args.output, word2index, embeddings_vectors)

    print("Saving mapping")
    save_oov_mapping(oov_vocabulary_map, args.mapping)
