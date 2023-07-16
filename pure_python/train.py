import pure_python.delta_rule.one_layer_net as delta_rule
import pure_python.gradient_descent.one_layer_net as gradient_descent
from datareader import DataReader
from pure_python.vector_ import Vector
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
num_epochs = 5

# Network params
input_channels = 1
input_height = 28
input_width = 28
num_classes = 6

one_layer_net = gradient_descent.OneLayerNet(input_height * input_width, num_classes)

train_generator = DataReader('../data/train', [input_height, input_width], True, input_channels, num_classes)\
    .get_generator()
test_generator = DataReader('../data/test', [input_height, input_width], False, input_channels, num_classes)\
    .get_generator()

print(f'Size of training set: {train_generator.get_data_size()}')
print(f'Size of testing set: {test_generator.get_data_size()}')

print(f'{datetime.now()} Start training...')
for epoch in range(num_epochs):
    print(f'{datetime.now()} Epoch number: {epoch + 1}')
    loss = 0
    for m in range(train_generator.get_data_size()):
        x, desired = train_generator.next()
        loss += one_layer_net.train(Vector(x, desired), learning_rate)
    print(f'loss = {(loss / train_generator.get_data_size()):6f}')
    train_generator.reset_pointer()
    train_generator.shuffle_data()


print(f'{datetime.now()} Start testing...')
passed = 0
for i in range(test_generator.get_data_size()):
    x, desired = test_generator.next()
    y = one_layer_net.test(Vector(x, desired))
    print(f'neurons outputs: {y}')

    d_max_idx = get_max_neuron_idx(desired)
    y_max_idx = get_max_neuron_idx(y)
    if y_max_idx == d_max_idx:
        passed += 1
    print(f'{d_max_idx} recognized as {y_max_idx}')

accuracy = passed / test_generator.get_data_size() * 100.0
print(f'Accuracy: {accuracy:.4f}%')
