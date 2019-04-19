from neural_net import NeuralNet
from datareader import DataReader
from vector_ import Vector
from datetime import datetime


def get_max_neuron_idx(neurons):
    max_idx = -1
    answer = -1
    for j in range(len(neurons)):
        if neurons[j] > answer:
            answer = neurons[j]
            max_idx = j
    return max_idx


# Learning params
learning_rate = 1e-6
num_epochs = 10

# Network params
input_channels = 1
input_height = 28
input_width = 28
num_classes = 6

one_layer_net = NeuralNet(input_height * input_width, num_classes)

train_dir = 'data/train'
test_dir = 'data/test'

train_generator = DataReader(train_dir, [input_height, input_width], True, input_channels, num_classes).get_generator()
test_generator = DataReader(test_dir, [input_height, input_width], False, input_channels, num_classes).get_generator()

print('Size of training set: {}'.format(train_generator.get_data_size()))
print('Size of testing set: {}'.format(test_generator.get_data_size()))

print("{} Start training...".format(datetime.now()))
for epoch in range(num_epochs):
    print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
    loss = 0
    for m in range(train_generator.get_data_size()):
        x, d = train_generator.next()
        loss += one_layer_net.train(Vector(x, d), learning_rate)
    print("loss = {}".format(loss / train_generator.get_data_size()))
    train_generator.reset_pointer()
    train_generator.shuffle_data()

passed = 0
for i in range(test_generator.get_data_size()):
    x, d = test_generator.next()
    y = one_layer_net.test(Vector(x, d))
    loss = 0

    d_max_idx = get_max_neuron_idx(d)
    y_max_idx = get_max_neuron_idx(y)
    if y_max_idx == d_max_idx:
        passed += 1
    print("{} recognized as {}".format(d_max_idx, y_max_idx))

accuracy = passed / test_generator.get_data_size() * 100.0
print("Accuracy: {:.4f}%".format(accuracy))
