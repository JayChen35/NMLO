import pandas as pd
import numpy as np
import random

def shuffle(X, y):
    assert(len(X) == len(y))
    idxs = [i for i in range(len(X))]
    random.shuffle(idxs)
    X_s = np.array([X[i] for i in idxs], dtype=X.dtype)
    y_s = np.array([y[i] for i in idxs], dtype=y.dtype)
    return X_s, y_s


def get_training_data():
    df = pd.read_csv('nmlo-contest-3/train.csv', sep=',',header=None)
    all_data = df.values[1:].astype(np.float32)
    X = all_data[:,1:4].astype(np.float32)
    y = all_data[:,4].astype(np.float32)
    X, y = shuffle(X, y)
    return X, y


def split(X, y, split_percent=0.1):
    X, y = shuffle(X, y)
    split_val = int(split_percent * len(X))
    return X[split_val:], y[split_val:], X[:split_val], y[:split_val]


def get_testing_data():
    df = pd.read_csv('nmlo-contest-3/test.csv', sep=',',header=None)
    all_data = df.values[1:].astype(np.float32)
    return all_data


def submit(preds, filename):
    file = open(filename, "w+")
    file.write("id,cases\n")
    for i, pred in enumerate(preds):
        file.write("{},{}\n".format(i, int(round(pred))))
    file.close()

