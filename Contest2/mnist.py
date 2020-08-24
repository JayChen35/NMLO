# MNIST Handwritten Digits Recognition
# Jason Chen, Period 4
# 01 June, 2020

import numpy as np
import time
from dataparser import train_test_split


def p_net(A, input_vec: np.ndarray, w_list: list, b_list: list):
    act = np.vectorize(A)
    a_not = input_vec
    for layer in range(1, len(w_list)):
        a_not = act(np.dot(a_not, w_list[layer]) + b_list[layer])
    return np.asarray(a_not)


def sigmoid(num: int or float):
    return 1/(1+np.exp(-1 * num))


def d_sigmoid(num: int or float):
    return sigmoid(num) * (1-sigmoid(num))


def error(output: np.ndarray, expected: np.ndarray):
    return 0.5 * (np.sum(np.square(np.subtract(expected, output))))


def back_propagation(w_list: list, b_list: list, training_set: np.ndarray or list, learning_rate: int or float):
    LAMBDA = learning_rate
    thingy = 0
    for x, y in training_set:
        y = [np.argmax(y)]
        thingy += 1
        if thingy % 10000 == 0:
            print(thingy)
        a_list = [np.array([x])]  # List of inputs
        dot_list = [None]
        for layer_num in range(1, len(w_list)):
            dot_L = np.dot(a_list[layer_num-1], w_list[layer_num]) + b_list[layer_num]
            dot_list.append(dot_L)
            a_L = sigmoid(dot_L)  # L denotes at a certain layer
            a_list.append(a_L)
        output = np.array([0 if x != y[0] else 1 for x in range(len(a_list[-1][0]))])
        delta_N = np.multiply(d_sigmoid(dot_list[-1]), np.subtract(output, a_list[-1]))  # N = index of last layer
        delta_list = [np.zeros((1, w_list[x].shape[1])) if x > 0 else None for x in range(len(w_list))]
        delta_list[-1] = delta_N
        for num in range(len(w_list)-2, 0, -1):  # Counting down from N-1, each layer in network
            delta_list[num] = np.multiply(d_sigmoid(dot_list[num]), np.dot(delta_list[num+1], w_list[num+1].T))
        for num in range(1, len(w_list)):  # Every layer in network, starting at 1 since index 0 is None
            b_list[num] = b_list[num] + LAMBDA*delta_list[num]
            w_list[num] = w_list[num] + LAMBDA*np.dot(a_list[num-1].T, delta_list[num])
    return w_list, b_list


def create_network(input_vec: tuple or list or np.ndarray):
    # input_vec is the conventional input scheme, e.g. 2-12-4-1 network = (2, 12, 4, 1)
    w_list = [None]
    b_list = [None]
    for x in range(len(input_vec)-1):
        w_list.append(np.random.uniform(-1, 1, (input_vec[x], input_vec[x+1])))
        b_list.append(np.random.uniform(-1, 1, (1, input_vec[x+1])))
    return w_list, b_list


def create_training_set(filename: str):
    training_set = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            temp = line.split(",")[1:]
            # Divide by 255 to put pixel values in range of sigmoid function
            training_set.append((np.asarray([int(x)/255 for x in temp[1:]]), np.array([int(temp[0])])))
    print(len(training_set), training_set[0].shape)
    print(1/0)
    return training_set
 

def save_wb_matrix(w_list: list, b_list: list):
    for layer in range(1, len(w_list)):
        np.savetxt("weights{}.csv".format(layer), np.asarray(w_list[layer]), delimiter=',', fmt='%s')
        np.savetxt("biases{}.csv".format(layer), np.asarray(b_list[layer]), delimiter=',', fmt='%s')


def eval_mnist(w_list, b_list, num_layers: int, dataset: list):
    num_correct = 0
    for x, y in dataset:
        eval = p_net(sigmoid, x, w_list, b_list)
        if np.argmax(eval) == np.argmax(y):
            num_correct += 1
    print("Accuracy: {}. Number correct/total: {}/{}.".format(
        num_correct/len(dataset), num_correct, len(dataset)))


def train_mnist(num_layers: int):
    # Parameters
    NUM_EPOCHS = 10000
    LAMBDA = 1e-03
    LOAD_WB_MATRIX = False
    if LOAD_WB_MATRIX:
        w_list = [None]
        b_list = [None]
        for n in range(1, num_layers+1):
            w_list.append(np.loadtxt("weights{}.csv".format(n), delimiter=','))
            b_list.append(np.loadtxt("biases{}.csv".format(n), delimiter=','))
        b_list = [np.expand_dims(b_list[x], axis=0) if x > 0 else None for x in range(len(b_list))]
    else:
        network_tuple = (784, 300, 100, 10)
        assert len(network_tuple)-1 == num_layers
        w_list, b_list = create_network(network_tuple)
    total_time = 0
    _, _, train_data, val_data = train_test_split()
    # training_set = create_training_set("mnist_train.csv")
    print("Validation accuracy:", eval_mnist(w_list, b_list, num_layers, val_data))
    for epoch in range(1, NUM_EPOCHS+1):
        start = time.perf_counter()
        w_list, b_list = back_propagation(w_list, b_list, train_data, LAMBDA)
        print("Backpropogated")
        time_elapsed = time.perf_counter() - start
        total_time += time_elapsed
        save_wb_matrix(w_list, b_list)
        print("Validation accuracy:", eval_mnist(w_list, b_list, num_layers, val_data))
        print("Epoch {} of {} complete. Time elapsed: {} seconds. Total time: {} seconds.".format(
            epoch, NUM_EPOCHS, round(time_elapsed, 2), round(total_time, 2)))


if __name__ == '__main__':
    train_mnist(3)
