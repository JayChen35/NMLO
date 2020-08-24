from nn import NN
from dataparser import train_test_split, submit

input_size, output_size, train_data, test_data = train_test_split(False, False, 0.1, True)
layers = [input_size, 64, 128, output_size]
print(layers)
# layers = [input_size, output_size]
lr = (0.001, 0.001)
transfer = "logistic"
brain = NN(layers, lr, transfer, 32)
brain.load('weights/weights4_final.npz')

print("Error 1:", brain.get_error(train_data))
print("Accuracy 1:", brain.get_correct(train_data))
print("Error 2:", brain.get_error(test_data))
print("Accuracy 2:", brain.get_correct(test_data))

submit(brain, 'submission.csv', True)
