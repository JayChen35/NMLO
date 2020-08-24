import numpy as np
import math
import random

# Transfer functions
def linear(x):
    return x

def ramp(x):
    return max(x, 0)

def logistic(x):
    return 1 / (1 + math.e ** -x)

def dlogistic(x, log=False):
    if log:
        return logistic(x) * (1 - logistic(x))
    return x * (1 - x)

def sigmoid(x):
    return 2 * logistic(x) - 1

def calc_error(pred, labels):
    final = pred[-1]
    assert(final.shape == labels.shape)
    return (1/2) * sum((final - labels) ** 2)

def get_correct(pred, labels):
    final = pred[-1]
    return np.argmax(final) == np.argmax(labels)

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

class NN:
    def __init__(self, layer_counts, lr, transfer, batch_size, initial_weights = None):
        self.layer_counts = layer_counts
        self.max_lr, self.min_lr = lr
        self.lr = self.max_lr
        self.batch_size = batch_size
        self.deltas = None
        # self.batch_size = batch_size
        if initial_weights is None:
            self.build_random()
        else:
            self.weights, self.biases = initial_weights
        transfer_map = {"logistic": (logistic, dlogistic)}
        self.transfer, self.dtrasnfer = transfer_map[transfer]


    def build_random(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_counts) - 1):
            inp, out = self.layer_counts[i], self.layer_counts[i + 1]
            self.weights.append(np.random.rand(inp, out) * 2 - 1)
            self.biases.append(random.random() * 2 - 1)


    def feedforward(self, inp):
        layers = [np.copy(inp)]
        curr = np.copy(inp)
        for i in range(len(self.layer_counts) - 1):
            W, B = self.weights[i], self.biases[i]
            curr = (curr.reshape(1, -1) @ W).reshape(-1,)
            # print("CURR")
            # print(curr + B)
            curr = self.transfer(curr + B)
            layers.append(np.copy(curr))
        return layers


    def backprop(self, layers, correct):
#        deltas = [None for i in range(len(self.weights))]
        dE_dn = None

        for L in reversed(range(len(self.weights))):
            if dE_dn is None:
                dE_do = (layers[-1] - correct)
            else:
                dE_do = (self.weights[L + 1] @ dE_dn.reshape(-1, 1)).reshape(-1,)
            do_dn = self.dtrasnfer(layers[L + 1])
            dE_dn = dE_do * do_dn
            dn_dw = layers[L]
            delta = dn_dw.reshape(-1, 1) @ dE_dn.reshape(1, -1)
            if self.deltas is None:
                self.deltas = [None for i in range(len(self.weights))]
            if self.deltas[L] is None:
                self.deltas[L] = delta
            else:
                self.deltas[L] += delta


    def update_weights(self):
        if self.deltas is None:
            return
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * self.deltas[i] * self.batch_size
        self.deltas = None

        # print(self.weights)

    def learn(self, train_data, test_data, num_epochs, save_interval, filename):
        for epoch in range(num_epochs):
            self.lr = translate(epoch, 0, num_epochs, self.max_lr, self.min_lr)
            print("Learning rate:", self.lr)
            if epoch % save_interval == 0 and save_interval != -1 and epoch != 0:
                self.save(filename.format(epoch))
            print("Epoch:", epoch)
            random.shuffle(train_data)
            for i, sample in enumerate(train_data):
                if i % 10000 == 0:
                    print(i)
                    # print("Accuracy:", self.get_correct(test_data))
                inputs, labels = sample
                layers = self.feedforward(inputs)
                self.backprop(layers, labels)
                if i % self.batch_size == 0:
                    self.update_weights()
            self.update_weights()
            print("Training Error:", self.get_error(train_data))
            print("Training Accuracy:", self.get_correct(train_data))
            print("Validation Error:", self.get_error(test_data))
            print("Validation Accuracy:", self.get_correct(test_data))


    def get_error(self, data):
        avg = 0
        for inp, out in data:
            pred = self.feedforward(inp)
            err = calc_error(pred, out)
            avg += err
        avg /= len(data)
        return avg

    def get_correct(self, data):
        avg = 0
        for inp, out in data:
            pred = self.feedforward(inp)
            correct = get_correct(pred, out)
            avg += 1 if correct else 0
        avg /= len(data)
        return avg

    def save(self, filename='weights.npz'):
        np.savez(filename, *self.weights)
    
    def load(self, filename='weights.npz'):
        container = np.load(filename)
        self.weights = [container[key] for key in container]


