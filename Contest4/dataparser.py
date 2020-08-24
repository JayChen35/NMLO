import pandas as pd
import numpy as np
import random
import cv2
import os
import tensorflow as tf

CLASS_LABELS = ['Apple Braeburn', 'Apple Granny Smith', 'Apricot', 'Avocado', 'Banana', 'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Cherry', 'Clementine', 'Corn', 'Cucumber Ripe', 'Grape Blue', 'Kiwi', 'Lemon', 'Limes', 'Mango', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Pear', 'Pepper Green', 'Pepper Red', 'Pineapple', 'Plum', 'Pomegranate', 'Potato Red', 'Raspberry', 'Strawberry', 'Tomato', 'Watermelon']

def image_to_numpy(filename):
    img = cv2.imread(filename)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = 255.0 - img
    img /= 255.0
    img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR)
    # img = img.reshape(*img.shape, 1)
    return img


def get_training_data(saved=False, use_preprocessed=False):
    filenames = "imgs.npy", "labels.npy"
    if use_preprocessed:
        filenames = "imgs_pre.npy", "labels_pre.npy"
    if saved:
        imgs = np.load(filenames[0])
        labels = np.load(filenames[1])
        label_names = CLASS_LABELS
        return imgs, labels, label_names
    base = "nmlo-contest-4/train/train/"
    labels = []
    label_names = []
    imgs = []
    for labelnum, folder in enumerate(os.listdir(base)):
        print(folder)
        path = os.path.join(base, folder)
        if not os.path.isdir(path):
            continue
        label_names.append(folder)
        for filename in os.listdir(path):
            filename = os.path.join(os.path.join(base, folder), filename)
            img = image_to_numpy(filename)
            imgs.append(img)
            labels.append(labelnum)
    imgs = np.array(imgs)
    labels = np.array(labels)
    # labels = one_hot(labels)
    print(label_names)
    print(labels.shape)
    np.save(filenames[0][:-4], imgs)
    np.save(filenames[1][:-4], labels)
    return imgs, labels, label_names


# Returns: input size, output size, training data, testing data
# def split(X, y, split_proportion=0.1):
#     idxs = [i for i in range(len(X))]
#     random.shuffle(idxs)
#     X_shuffled = np.array([X[i] for i in idxs])
#     y_shuffled = np.array([y[i] for i in idxs])
#     split_val = int(split_proportion * len(X))
#     return X_shuffled[split_val:], y_shuffled[split_val:], X_shuffled[:split_val], y_shuffled[:split_val]


def split(X, y, split_proportion):
    data = list(zip(X,y))
    all_labels = set()
    for x, y in data:
        all_labels.add(y)

    num_labels_validation = int(split_proportion * len(data)) // len(all_labels)
    random.shuffle(data)
    val_data = []
    train_data = []
    counts = {}
    for x, y in data:
        label = y
        if label not in counts:
            counts[label] = 0
        if counts[label] < num_labels_validation:
            val_data.append((x,y))
            counts[label] += 1
        else:
            train_data.append((x,y))

    X_train, y_train = np.array([i[0] for i in train_data]), np.array([i[1] for i in train_data])
    X_val, y_val = np.array([i[0] for i in val_data]), np.array([i[1] for i in val_data])
    print(len(train_data), len(val_data))    
    return X_train, y_train, X_val, y_val


def get_testing_data():
    base = "nmlo-contest-4/test/test"
    imgs = []
    ids = []
    for filename in os.listdir(base):
        id = int(filename[:filename.index(".")])
        img = image_to_numpy(os.path.join(base, filename))
        imgs.append(img)
        ids.append(id)
    imgs = np.array(imgs)
    return imgs, ids, CLASS_LABELS

