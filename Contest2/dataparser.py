import pandas as pd
import numpy as np
import random
from preprocess import preprocess

def one_hot(a):
    assert(len(a.shape) == 1)
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b

def split(data, split_proportion):
    all_labels = np.zeros(data[0][1].shape)
    for x, y in data:
        all_labels = all_labels + y
    
    num_labels_validation = int(split_proportion * len(data)) // len(all_labels)
    random.shuffle(data)
    val_data = []
    train_data = []
    counts = {}
    for x, y in data:
        label = np.argmax(y)
        if label not in counts:
            counts[label] = 0
        if counts[label] < num_labels_validation:
            val_data.append((x,y))
            counts[label] += 1
        else:
            train_data.append((x,y))

    print(len(train_data), len(val_data))    
    return train_data, val_data
    # TODO: DO THIS
    # TODO: IMPLEMENT ADAM


# Returns: input size, output size, training data, testing data
def train_test_split(use_saved=False, use_preprocessed=False, split_proportion=0.1, inherent_bias=True):
    if use_saved and use_preprocessed:
        pixels = np.load('preprocessed_pixels.npy')
        labels = np.load('preprocessed_labels.npy')
        inp_size = pixels.shape[1]
    else:
        df = pd.read_csv('nmlo-contest-2/train/train.csv', sep=',',header=None)
        all_data = df.values
        inp_size = all_data.shape[1] - 2
        pixels = all_data[1:,2:].astype(np.float32)
        pixels /= 255
        labels = all_data[1:, 1].astype(np.uint8)
        labels = one_hot(labels)
        if use_preprocessed:
            pixels = preprocess(pixels)
            np.save('preprocessed_pixels.npy', pixels)
            np.save('preprocessed_labels.npy', labels)
            print(pixels.shape)
    if inherent_bias:
        biases = np.ones((len(pixels),1))
        pixels = np.hstack((pixels, biases))
        assert(pixels.shape[1] in [785, 289])
        inp_size += 1
    out_size = labels.shape[1]
    data = list(zip(pixels, labels))
    random.shuffle(data)
    train_data, val_data = split(data, split_proportion)
    print(inp_size, out_size)
    return inp_size, out_size, train_data, val_data


def submit(brain, filename="submission.csv", inherent_bias=False):
    df = pd.read_csv('nmlo-contest-2/test/test.csv', sep=',',header=None)
    all_data = df.values
    all_data = all_data[1:]
    output = []
    for i in range(len(all_data)):
        sample = all_data[i]
        id = sample[0]
        pixels = sample[1:].astype(np.float32)
        pixels /= 255
        # print(pixels.shape)
        if inherent_bias:
            pixels = np.append(pixels, np.array([1]))
        # print(pixels.shape)
        pred = brain.feedforward(pixels)[-1]
        guess = np.argmax(pred)
        output.append((id, guess))
    output_np = np.zeros((len(output), 2))
    for i in range(len(output)):
        output_np[i][0] = output[i][0]
        output_np[i][1] = output[i][1]
    output_np = output_np.astype(int)
    np.savetxt(filename, output_np, fmt='%i', delimiter=",")

