import tensorflow as tf
import numpy as np
import pickle
import pandas as pd

pickle_in = open("music_genres_dataset.pkl", "rb")
dataset = pd.DataFrame.from_dict(pickle.load(pickle_in))

dataset = dataset.sample(frac=1).reset_index(drop=True)
xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='biases')


def sample(data):
    dataset = data.sample(frac=1).reset_index(drop=True)
    split = 10000
    trainBatch = dataset[0:split - 1]["data"]
    trainLabels = dataset[0:split - 1]["labels"]
    testBatch = dataset[split:]["data"]
    testLabels = dataset[split:]["labels"]
    return trainBatch, testBatch, trainLabels, testLabels


def shallownn(x):
    x = tf.reshape(x, [-1, 80, 80, 1])
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=16,
        kernel_size=[10, 23],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        activation=tf.nn.leaky_relu,
        name='conv1'
    )
    h_pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[1, 20],
        strides=2,
        name='pool1'
    )
    conv2 = tf.layers.conv2d(
        inputs=x,
        filters=16,
        kernel_size=[21, 20],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        activation=tf.nn.leaky_relu,
        name='conv2'
    )
    h_pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[20, 1],
        strides=2,
        name='pool2'
    )
    h_pool2 = tf.reshape(x, [-1, 40, 31, 16])
    merge_layer = tf.concat([h_pool1, h_pool2], axis=3)

    dropout_layer = tf.layers.dropout(merge_layer, rate=0.1, name="dropout_layer")

    y_1 = tf.layers.dense(inputs=dropout_layer, units=200, activation=tf.nn.relu,
                          kernel_initializer=xavier_initializer, bias_initializer=xavier_initializer, trainable=True,
                          name="fc1")

    y_hat = tf.layers.dense(inputs=y_1, units=10, kernel_initializer=xavier_initializer,
                            bias_initializer=xavier_initializer, trainable=True, name="fc3")

    return y_hat


tf.reset_default_graph()
g = tf.get_default_graph()
with g.as_default():
    with tf.Session() as sess:
        with tf.variable_scope("inputs"):
            x = tf.placeholder(tf.float32, [None, 80 * 80])
            y = tf.placeholder(tf.float32, [None, 10])

        train, test, train_l, test_l = sample(dataset)
        y_hat = shallownn(x)
