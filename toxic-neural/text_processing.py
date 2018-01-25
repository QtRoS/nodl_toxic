import re
import numpy as np


def tokenize(text, lowercase=True):
    if lowercase:
        text = text.lower()
    delimeter = "([?\\/.,`~!@#4%^&*()-+\[\]{}<>'\"]*[ \s\n\t\r]+)"
    tokens = re.split(delimeter, text + " ")
    stripped_tokens = map(str.strip, tokens)
    noempty_tokens = filter(bool, stripped_tokens)
    return list(noempty_tokens)


def text_to_sequence(text, word2index, maxlen):
    tokens = tokenize(text)
    indices = [word2index[token] for token in tokens if token in word2index]
    sequence = np.ones([maxlen], dtype=np.int) * (len(word2index) - 1)
    size = min(len(indices), maxlen)
    sequence[:size] = indices[:size]
    return sequence