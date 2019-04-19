import tensorflow as tf


def fc(x, num_in, num_out, name, act_func=None, initializer=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  initializer=initializer)
        biases = tf.get_variable('biases', [num_out],
                                 initializer=initializer)

        # Matrix multiply weights and inputs and add bias
        net = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if act_func is not None:
            # Apply ReLu non linearity
            return act_func(net)

        return net
