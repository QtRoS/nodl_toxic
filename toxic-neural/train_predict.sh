#!/bin/bash
python3 oov_embeddings.py data/train.csv data/test.csv data/crawl-300d-2M.vec data/output.vec data/oov-mapping.json
for i in 1 2 3 4 5 6 7 8 9 10
do
    python3 model_train.py data/train.csv data/test.csv data/sample_submission.csv data/output.vec model "$i" predictions
done
for i in 0 1 2 3 4 5 6 7 8 9
do
    python3 model_predict.py data/train.csv data/test.csv data/output.vec "model/${i}.h5" "predictions/${i}.npy"
done
python3 build_submission.py predictions data/sample_submission.csv submission.csv