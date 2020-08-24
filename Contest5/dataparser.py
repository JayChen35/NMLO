  
import pandas as pd
import numpy as np
import random

def shuffle(X, y):
    assert(len(X) == len(y))
    idxs = [i for i in range(len(X))]
    random.shuffle(idxs)
    X_shuffled = np.array([X[i] for i in idxs], dtype=X.dtype)
    y_shuffled = np.array([y[i] for i in idxs], dtype=y.dtype)
    return X_shuffled, y_shuffled


def get_training_data():
    df = pd.read_csv('nmlo-contest-5/train.csv', sep=',',header=None)
    all_data = df.values
    sentences = all_data[1:,0].astype(object)
    labels = all_data[1:, 1].astype(int)
    sentences, labels = shuffle(sentences, labels)
    return sentences, labels


# Returns: input size, output size, training data, testing data
def split(X, y, split_proportion=0.1):
    X, y = shuffle(X, y)
    split_val = int(split_proportion * len(X))
    return X[split_val:], y[split_val:], X[:split_val], y[:split_val]


def get_testing_data():
    df = pd.read_csv('nmlo-contest-5/test.csv', sep=',',header=None)
    all_data = df.values
    sentences = all_data[1:,1].astype(object)
    ids = all_data[1:, 0].astype(int)
    return ids, sentences

