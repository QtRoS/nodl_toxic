#!/bin/bash
python3 validate_predictions.py data/train.csv data/test.csv data/output.vec models val 0
python3 validate_predictions.py data/train.csv data/test.csv data/output.vec models val 1
python3 validate_predictions.py data/train.csv data/test.csv data/output.vec models val 2
python3 validate_predictions.py data/train.csv data/test.csv data/output.vec models val 3
python3 validate_predictions.py data/train.csv data/test.csv data/output.vec models val 4
python3 validate_predictions.py data/train.csv data/test.csv data/output.vec models val 5
python3 validate_predictions.py data/train.csv data/test.csv data/output.vec models val 6
python3 validate_predictions.py data/train.csv data/test.csv data/output.vec models val 7
python3 validate_predictions.py data/train.csv data/test.csv data/output.vec models val 8
python3 validate_predictions.py data/train.csv data/test.csv data/output.vec models val 9
