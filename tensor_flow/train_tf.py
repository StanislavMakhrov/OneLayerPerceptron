import os
from tensorflow import keras
from datareader import DataReader
from datetime import datetime
import numpy as np
from tensor_flow.vector import Vector


def get_max_neuron_idx(neurons):
    max_idx = -1
    answer = -1
    for j in range(len(neurons)):
        if neurons[j] > answer:
            answer = neurons[j]
            max_idx = j
    return max_idx


# Learning params
learning_rate = 0.01
num_epochs = 10

# Network params
input_channels = 1
input_height = 28
input_width = 28
num_classes = 6
save_histogram = False

log_dir = "../logs/" + datetime.now().strftime("%Y-%m-%d.%H-%M-%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

train_dir = '../data/train'
test_dir = '../data/test'

train_generator = DataReader(train_dir, [input_height, input_width], True, input_channels, num_classes).get_generator()
test_generator = DataReader(test_dir, [input_height, input_width], False, input_channels, num_classes).get_generator()

((x_train, y_train), (x_test, y_test)) = (([], []), ([], []))

for m in range(train_generator.get_data_size()):
    vector = Vector(*train_generator.next(one_hot_encoding=False))
    x_train.append(vector.get_x())
    y_train.append(vector.get_desired_outputs())

x_train = np.array(x_train)
y_train = np.array(y_train).reshape(-1)

for m in range(test_generator.get_data_size()):
    vector = Vector(*test_generator.next(one_hot_encoding=False))
    x_test.append(vector.get_x())
    y_test.append(vector.get_desired_outputs())

x_test = np.array(x_test)
y_test = np.array(y_test).reshape(-1)

model = keras.models.Sequential([
    keras.layers.Dense(num_classes, activation='sigmoid'),
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
    metrics=['accuracy']
)

print(f'{datetime.now()} Start training...')
model.fit(
    x_train,  # input
    y_train,  # output
    verbose=1,
    epochs=num_epochs,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback],
)


# Test the model on the entire test set
print(f'{datetime.now()} Start testing...')
passed = 0
for i in range(len(x_test)):
    predictions = model(x_test[i:i+1]).numpy()
    print(predictions)

#     prediction = sess.run(tf.nn.sigmoid(model.output), feed_dict={x: xs})
#
#     d_max_idx = get_max_neuron_idx(list(ds.reshape(ds[0].shape)))
#     y_max_idx = get_max_neuron_idx(prediction[0])
#     if y_max_idx == d_max_idx:
#         passed += 1
#     print("{} recognized as {}".format(d_max_idx, y_max_idx))
#
# accuracy = passed / test_generator.get_data_size() * 100.0
# print("Accuracy: {:.4f}%".format(accuracy))
