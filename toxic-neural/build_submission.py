from argparse import ArgumentParser
from os import listdir, path
import numpy
import pandas as pd


def get_args():
    parser = ArgumentParser()
    parser.add_argument("predictions_dir")
    parser.add_argument("submission")
    parser.add_argument("output")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print("Reading predictions")
    predictions = []
    print("Calculating result")
    for fname in listdir(args.predictions_dir):
        predictions.append(numpy.load(path.join(args.predictions_dir, fname)))
    result = numpy.ones(predictions[0].shape)
    for prediction in predictions:
        result *= prediction
    result **= (1 / len(predictions))
    result **= 1.4

    print("Reading submission")
    submission = pd.read_csv(args.submission)

    targets = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    submission[targets] = result

    print("Saving prediction")
    submission.to_csv(args.output, index=None)