import random
import numpy as np
from nn import NN, calc_error
from dataparser import train_test_split
from preprocess import preprocess
import time


input_size, output_size, train_data, test_data = train_test_split(False, False, 0.1, True)
#train_data = train_data[len(train_data) // 2:]
print("Parsed data")

assert(input_size in [784, 785, 288, 289] and output_size == 10)

layers = [input_size, 128, 64, output_size]
print(layers)
# layers = [input_size, output_size]
lr = (0.001, 0.0005)
transfer = "logistic"
batch_size = 4
brain = NN(layers, lr, transfer, batch_size)

print("Error", brain.get_error(test_data))
print("Training")
print(len(train_data))

start = time.time()
brain.learn(train_data, test_data, 100, 5, "weights/weights12864_{}.npz")
print("Took:", time.time() - start)
print("Error", brain.get_error(test_data))
print("Accuracy", brain.get_correct(test_data))

brain.save("weights/weights12864_final.npz")
