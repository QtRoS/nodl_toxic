from tqdm import tqdm
import numpy as np


def read_embeddings(fname, encoding='utf-8'):
    with open(fname, 'r', encoding=encoding) as src:
        header = src.readline()
        wordcount, vectorsize = map(int, header.split())
        word2index = {}
        vectors = np.zeros([wordcount, vectorsize])
        for i in tqdm(range(wordcount)):
            row = src.readline().split()
            if len(row) != vectorsize + 1:
                continue
            word = row[0]
            vector = np.array(list(map(float, row[1:])))
            word2index[word] = i
            vectors[i, :] = vector
    return word2index, vectors


def save_embeddings(fname, word2index, vectors, encoding='utf-8'):
    with open(fname, 'w', encoding=encoding) as target:
        target.write('{0} {1}\n'.format(len(word2index), vectors.shape[1]))
        for word, index in tqdm(word2index.items(), total=len(word2index)):
            vector = vectors[index]
            vector_str = ' '.join(map(str, vector))
            target.write('{0} {1}\n'.format(word, vector_str))


def cutten_embeddings(wordset, word2index, vectors):
    word2index_cut = {}
    for word in tqdm(sorted(wordset)):
        if word not in word2index:
            continue
        word2index_cut[word] = len(word2index_cut)
    vectors_cut = np.zeros([len(word2index_cut) + 1, vectors.shape[1]])
    for word, index in tqdm(word2index_cut.items()):
        vectors_cut[index, :] = vectors[word2index[word]]
    return word2index_cut, vectors_cut
