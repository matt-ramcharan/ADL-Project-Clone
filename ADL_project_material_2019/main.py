import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import utils

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
    trainBatch  = list(np.row_stack(dataset[0:split - 1]["data"].values))
    trainBatch  = np.array(list(map(utils.melspectrogram, trainBatch)))

    trainLabels = pd.get_dummies(dataset[0:split - 1]["labels"]).values

    testBatch   = np.row_stack(dataset[split:]["data"].values)
    testBatch  = np.array(list(map(utils.melspectrogram, testBatch)))

    testLabels  = pd.get_dummies(dataset[split:]["labels"]).values
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
    h_pool2 = tf.reshape(h_pool2, [-1, 40, 31, 16])
    merge_layer = tf.concat([h_pool1, h_pool2], axis=3)

    dropout_layer = tf.layers.dropout(merge_layer, rate=0.1, name="dropout_layer")

    y_1 = tf.layers.dense(inputs=dropout_layer, units=200, activation=tf.nn.relu,
                          kernel_initializer=xavier_initializer, bias_initializer=xavier_initializer, trainable=True,
                          name="fc1")

    y_1_soft = tf.nn.softmax(y_1)

    y_hat = tf.layers.dense(inputs=y_1_soft, units=10, kernel_initializer=xavier_initializer,
                            bias_initializer=xavier_initializer, trainable=True, name="fc3")

    return y_hat


tf.reset_default_graph()
g = tf.get_default_graph()
with g.as_default():
    with tf.Session() as sess:
        train, test, train_l, test_l = sample(dataset)
        # print(train_l)

        train_batch_size = 1
        test_batch_size = 1 

        train_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(train, tf.float32),
                    tf.cast(train_l, tf.float32)
                )
                ).batch(train_batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(test, tf.float32),
                    tf.cast(test_l, tf.int32)
                )
                ).batch(test_batch_size)

        iterator = train_dataset.make_one_shot_iterator()

        with tf.variable_scope("inputs"):
            x, y = iterator.get_next()
            # x = tf.placeholder(tf.float32, [None, 80, 80])
            # y = tf.placeholder(tf.float32, [None, 10])

            y_hat = shallownn(x);

            print(y.dtype)
            print(y_hat.dtype)

            cross_entropy = tf.keras.backend.categorical_crossentropy(y_hat, y)
            optimiser = tf.train.AdamOptimizer().minimize(cross_entropy)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for _ in range(100):
            print(_)
            sess.run(optimiser)
            # sess.run(optimiser) 
        

        y_hat = shallownn(x)
