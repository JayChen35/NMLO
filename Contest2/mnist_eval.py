# MNIST Handwritten Digits Recognition Network Evaluation
# Jason Chen, Period 4
# 01 June, 2020

import numpy as np


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
        for line in f:
            temp = line.split(",")
            training_set.append((np.asarray([int(x)/255 for x in temp[1:]]), np.array([int(temp[0])])))
    return training_set


def eval_mnist(num_layers: int):
    # Parameters
    w_list = [None]
    b_list = [None]
    for n in range(1, num_layers+1):
        w_list.append(np.loadtxt("weights{}.csv".format(n), delimiter=','))
        b_list.append(np.loadtxt("biases{}.csv".format(n), delimiter=','))
    b_list = [np.expand_dims(b_list[x], axis=0) if x > 0 else None for x in range(len(b_list))]
    testing_set = create_training_set("mnist_test.csv")
    num_correct = 0
    for x, y in testing_set:
        eval = p_net(sigmoid, x, w_list, b_list)
        if np.argmax(eval) == y[0]:
            num_correct += 1
    print("Accuracy: {}. Number correct/total: {}/{}.".format(
        num_correct/len(testing_set), num_correct, len(testing_set)))


if __name__ == '__main__':
    eval_mnist(3)
