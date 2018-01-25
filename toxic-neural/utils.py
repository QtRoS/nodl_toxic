from itertools import chain
from text_processing import tokenize


def get_wordset(texts):
    return set(chain(*map(tokenize, texts)))