#!/bin/python
import subprocess

subprocess.check_call(['python3', 'oov_embeddings.py', 'data/train.csv', 'data/test.csv', 'data/crawl-300d-2M.vec', 'data/output.vec', 'data/oov-mapping.json',])
for i in range(1, 11):
    subprocess.check_call(['python3', 'model_train.py', 'data/train.csv', 'data/test.csv', 'data/sample_submission.csv', 'data/output.vec', 'model', str(i), 'predictions',])
for i in range(0, 10):
    subprocess.check_call(['python3', 'model_predict.py', 'data/train.csv', 'data/test.csv', 'data/output.vec', 'model/{}.h5'.format(i), 'predictions/{}.npy'.format(i),])
subprocess.check_call(['python3', 'build_submission.py', 'predictions', 'data/sample_submission.csv', 'submission.csv',])