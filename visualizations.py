import numpy as np
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import pandas as pd


def topn_features(clf, n):
    weights = np.abs(np.stack([
        estimator.coef_[0]
        for estimator in clf.steps[1][1].estimators
    ]).max(axis=0))
    word_tfidf = clf.steps[0][1].transformer_list[0][1].steps[1][1]
    words = np.array(word_tfidf.get_feature_names())
    chars_tfidf = clf.steps[0][1].transformer_list[1][1].steps[1][1]
    chars = np.array(chars_tfidf.get_feature_names())
    word_weights = weights[0:len(words)]
    char_weights = weights[len(words): len(words) + len(chars)]
    sorted_word_weights_indices = word_weights.argsort()[::-1]
    sorted_char_weights_indices = char_weights.argsort()[::-1]
    top_words = words[sorted_word_weights_indices[:n]]
    top_words_weights = word_weights[sorted_word_weights_indices[:n]]
    top_words_dict = OrderedDict([
        (word, weight)
        for word, weight in zip(top_words, top_words_weights)
    ])
    top_chars = chars[sorted_char_weights_indices[:n]]
    top_chars_weights = char_weights[sorted_char_weights_indices[:n]]
    top_chars_dict = OrderedDict([
        (char, weight)
        for char, weight in zip(top_chars, top_chars_weights)
    ])
    return top_words_dict, top_chars_dict


def confusion_matrix(clf, X, y, do_fit=True):
    X = np.array(X)
    y = np.array(y)

    assert len(y.shape) == 2
    assert y.shape[1] == 1

    idx = np.arange(len(X), dtype=np.int)
    train_idx, test_idx, _, _ = train_test_split(idx,
                                                 y[:, 0],
                                                 stratify=y[:, 0])
    if do_fit:
        clf.fit(X[train_idx], y[train_idx])

    prediction = clf.predict(X[test_idx])[:, 0]
    test = y[test_idx, 0]

    true_positive_count = np.logical_and(test == 1, prediction == 1).sum()
    true_negative_count = np.logical_and(test == 0, prediction == 0).sum()
    false_positive_count = np.logical_and(test == 0, prediction == 1).sum()
    false_negative_count = np.logical_and(test == 1, prediction == 0).sum()

    negative_count = false_positive_count + true_negative_count
    positive_count = false_negative_count + true_positive_count

    false_positive_rate = false_positive_count / negative_count
    false_negative_rate = false_negative_count / positive_count
    true_positive_rate = true_positive_count / positive_count
    true_negative_rate = true_negative_count / negative_count

    matrix = np.array([
        [true_negative_rate, false_positive_rate],
        [false_negative_rate, true_positive_rate]
    ])

    return pd.DataFrame(matrix) \
        .rename({0: "predicted negative", 1: "predicted positive"}, axis=1) \
        .rename({0: "negative", 1: "positive"}, axis=0)
