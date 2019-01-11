import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import utils
from math import ceil
import os
import random

pickle_in_a = open("music_genres_dataset_aug.pkl", "rb")
dataset_a = pd.DataFrame.from_dict(pickle.load(pickle_in_a))

pickle_in = open("music_genres_dataset.pkl", "rb")
dataset = pd.DataFrame.from_dict(pickle.load(pickle_in))

logdir = '{cwd}/logs/log'.format(cwd=os.getcwd())

xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weights')


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='biases')




def sample(data, data_a):
    # data_a = pd.DataFrame([[row.get("data"), row.get("labels"), row.get("track_id")]  for idx, row in data_a.iterrows() if idx % 9 != 0],
    #                     columns=["data", "labels", "track_id"])[0:40000]

    groups = [data for _, data in data.groupby('track_id')]
    groups_a = [data for _, data in data_a.groupby('track_id')]

    random.shuffle(groups)
    random.shuffle(groups_a)

    dataset = pd.concat(groups).reset_index(drop=True)
    dataset_a = pd.concat(groups).reset_index(drop=True)

    trainBatch  = list(np.row_stack(dataset["data"].values))
    trainBatch  = np.array(list(map(utils.melspectrogram, trainBatch)), dtype=np.float32)

    trainLabels = np.array(pd.get_dummies(dataset_a["labels"]).values, dtype=np.float32)

    testBatch   = np.row_stack(dataset["data"].values)
    testBatch  = np.array(list(map(utils.melspectrogram, testBatch)), dtype=np.float32)

    testLabels  = np.array(pd.get_dummies(dataset["labels"]).values, dtype=np.float32)

    return trainBatch, testBatch, trainLabels, testLabels

leaky_relu = lambda x:tf.nn.leaky_relu(x, alpha=0.3)


def deepnn(x):
    x = tf.reshape(x, [-1, 80, 80, 1])

    conv1_1 = tf.layers.conv2d(
        inputs=x,
        filters=16,
        kernel_size=[10, 23],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        activation=leaky_relu,
        name='conv1_1'
    )

    conv1_1 = leaky_relu(tf.contrib.layers.batch_norm(conv1_1))

    h_pool1_1 = tf.layers.max_pooling2d(
        inputs=conv1_1,
        pool_size=[2, 2],
        strides=[2, 2],
        name='pool1_1'
    )
    conv2_1 = tf.layers.conv2d(
        inputs=x,
        filters=16,
    kernel_size=[21, 10],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        activation=leaky_relu,
        name='conv2_1'
    )

    conv2_1 = leaky_relu(tf.contrib.layers.batch_norm(conv2_1))

    h_pool2_1 = tf.layers.max_pooling2d(
        inputs=conv2_1,
        pool_size=[2, 2],
        strides=[2, 2],
        name='pool2_1'
    )

    conv1_2 = tf.layers.conv2d(
        inputs=h_pool1_1,
        filters=32,
        kernel_size=[5, 11],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        activation=leaky_relu,
        name='conv1_2'
    )
    conv1_2 = leaky_relu(tf.contrib.layers.batch_norm(conv1_2))
    h_pool1_2 = tf.layers.max_pooling2d(
        inputs=conv1_2,
        pool_size=[2, 2],
        strides=[2, 2],
        name='pool1_2'
    )
    conv2_2 = tf.layers.conv2d(
        inputs=h_pool2_1,
        filters=32,
        kernel_size=[10, 5],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        activation=leaky_relu,
        name='conv2_2'
    )
    conv2_2 = leaky_relu(tf.contrib.layers.batch_norm(conv2_2))
    h_pool2_2 = tf.layers.max_pooling2d(
        inputs=conv2_2,
        pool_size=[2, 2],
        strides=[2, 2],
        name='pool2_2'
    )

    conv1_3 = tf.layers.conv2d(
        inputs=h_pool1_2,
        filters=64,
        kernel_size=[3, 5],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        activation=leaky_relu,
        name='conv1_3'
    )
    conv1_3 = leaky_relu(tf.contrib.layers.batch_norm(conv1_3))
    h_pool1_3 = tf.layers.max_pooling2d(
        inputs=conv1_3,
        pool_size=[2, 2],
        strides=[2, 2],
        name='pool1_3'
    )
    conv2_3 = tf.layers.conv2d(
        inputs=h_pool2_2,
        filters=64,
        kernel_size=[5, 3],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        activation=leaky_relu,
        name='conv2_3'
    )
    conv2_3 = leaky_relu(tf.contrib.layers.batch_norm(conv2_3))
    h_pool2_3 = tf.layers.max_pooling2d(
        inputs=conv2_3,
        pool_size=[2, 2],
        strides=[2, 2],
        name='pool2_3'
    )

    conv1_4 = tf.layers.conv2d(
        inputs=h_pool1_3,
        filters=128,
        kernel_size=[2, 4],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        activation=leaky_relu,
        name='conv1_4'
    )
    conv1_4 = leaky_relu(tf.contrib.layers.batch_norm(conv1_4))
    h_pool1_4 = tf.layers.max_pooling2d(
        inputs=conv1_4,
        pool_size=[1, 5],
        strides=[1, 5],
        name='pool1_4'
    )
    conv2_4 = tf.layers.conv2d(
        inputs=h_pool2_3,
        filters=128,
        kernel_size=[4, 2],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        activation=leaky_relu,
        name='conv2_4'
    )
    conv2_4 = leaky_relu(tf.contrib.layers.batch_norm(conv2_4))
    h_pool2_4 = tf.layers.max_pooling2d(
        inputs=conv2_4,
        pool_size=[5, 1],
        strides=[5, 1],
        name='pool2_4'
    )

    print(h_pool1_4.get_shape())
    print(h_pool2_4.get_shape())
    h_pool1_4 = tf.reshape(h_pool1_4, [-1, 2560])
    h_pool2_4 = tf.reshape(h_pool2_4, [-1, 2560])
    merge_layer = tf.concat([h_pool1_4, h_pool2_4], axis=1)
    dropout_layer = tf.layers.dropout(merge_layer, rate=0.25, name="dropout_layer")
    y_1 = tf.layers.dense(inputs=dropout_layer, units=200, activation=tf.nn.relu,
                          kernel_initializer=xavier_initializer, bias_initializer=xavier_initializer, trainable=True,
                          name="fc1")

    y_1_soft = tf.nn.softmax(y_1)

    y_hat = tf.layers.dense(inputs=y_1_soft, units=10, kernel_initializer=xavier_initializer,
                            bias_initializer=xavier_initializer, trainable=True, name="fc3")
    return y_hat


def shallownn(x):
    x = tf.reshape(x, [-1, 80, 80, 1])
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=16,
        kernel_size=[10, 23],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        name='conv1'
    )

    conv1 = leaky_relu(tf.contrib.layers.batch_norm(conv1))


    h_pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[1, 20],
        strides=[1, 20],
        name='pool1'
    )
    conv2 = tf.layers.conv2d(
        inputs=x,
        filters=16,
        kernel_size=[21, 20],
        padding='same',
        use_bias=False,
        kernel_initializer=xavier_initializer,
        activation=leaky_relu,
        name='conv2'
    )

    conv2 = leaky_relu(tf.contrib.layers.batch_norm(conv2))

    h_pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[20, 1],
        strides=[20, 1],
        name='pool2'
    )
    h_pool1 = tf.reshape(h_pool1, [-1, 5120])
    h_pool2 = tf.reshape(h_pool2, [-1, 5120])
    merge_layer = tf.concat([h_pool1, h_pool2], axis=1)
    dropout_layer = tf.layers.dropout(merge_layer, rate=0.1, name="dropout_layer")
    y_1 = tf.layers.dense(inputs=dropout_layer, units=200,
                          kernel_initializer=xavier_initializer, bias_initializer=xavier_initializer, trainable=True,
                          name="fc1")

    # y_1_soft = tf.nn.softmax(y_1)

    y_hat = tf.layers.dense(inputs=y_1, units=10, kernel_initializer=xavier_initializer,
                            bias_initializer=xavier_initializer, trainable=True, name="fc3")



    return y_hat

def main(_):
    tf.reset_default_graph()
    g = tf.get_default_graph()
    with g.as_default():
        with tf.Session() as sess:

            print("Starting data loading")
            train, test, train_l, test_l = sample(dataset, dataset_a)
            print("Finished data loading")

            learning_rate = 0.00005

            train_batch_size = 64
            test_batch_size  = 15

            total_iteration_amount = 500000
            epoch_am = ceil(total_iteration_amount / len(train))

            train_placeholder = tf.placeholder(train.dtype, train.shape)
            test_placeholder = tf.placeholder(test.dtype, test.shape)
            train_l_placeholder = tf.placeholder(train_l.dtype, train_l.shape)
            test_l_placeholder = tf.placeholder(test_l.dtype, test_l.shape)

            train_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    train_placeholder,
                    train_l_placeholder
                )
            ).batch(train_batch_size).repeat(epoch_am).shuffle(len(train), seed=0)

            test_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    test_placeholder,
                    test_l_placeholder
                )
            ).batch(test_batch_size)


            iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                       train_dataset.output_shapes)

            train_init_op = iterator.make_initializer(train_dataset)
            test_init_op  = iterator.make_initializer(test_dataset)


            print('Running ' + str(total_iteration_amount) + ' iteration over '
                + str(epoch_am) + ' epochs')



            with tf.variable_scope("inputs"):
                x, y = iterator.get_next()

                y_hat = shallownn(x)

            with tf.variable_scope("cross_entropy"):
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

            l1_regularizer = tf.contrib.layers.l1_regularizer(
               scale=0.0001, scope=None
            )
            weights = tf.trainable_variables() # all vars of your graph
            regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
            
            regularized_loss = cross_entropy + regularization_penalty # this loss needs to be minimized


            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimiser = tf.train.AdamOptimizer(learning_rate).minimize(regularized_loss)

            with tf.name_scope("acc_raw"):
                acc_raw = tf.reduce_mean(
                          tf.cast(
                          tf.math.equal(
                          tf.argmax(y_hat, axis=1),
                          tf.argmax(y, axis=1)),
                          tf.float32))

            with tf.name_scope("acc-max"):
                acc_max = tf.cast(
                          tf.math.equal(
                          tf.argmax( tf.reduce_sum(y_hat, axis=0)),
                          tf.argmax( tf.reduce_mean(y, axis=0))),
                          tf.float32)#, [1]),
                          # tf.constant([15]))

            with tf.name_scope("acc-maj"):
                acc_maj = tf.cast(
                          tf.math.equal(
                          tf.argmax( tf.reduce_sum(tf.one_hot(tf.argmax(y_hat, axis=1), 10), axis=0)),
                          tf.argmax( tf.reduce_mean(y, axis=0))),
                          tf.float32)#, [1]),
                          # tf.constant([15]))

            loss_summary = tf.summary.scalar('Loss', cross_entropy)
            acc_summary  = tf.summary.scalar('acc_raw', acc_raw)

            # train_data_sum = tf.summary.audio('Input_Train_Audio', train, 22050)
            # test_data_sum = tf.summary.audio('Input_Test_Audio', test, 22050)

            train_summary = tf.summary.merge([acc_summary, loss_summary])
            test_summary = tf.summary.merge([acc_summary, loss_summary])

            summary_writer = tf.summary.FileWriter(logdir+'_train', sess.graph, flush_secs=5)
            summary_test_writer = tf.summary.FileWriter(logdir+'_test', sess.graph, flush_secs=5)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            print(x.shape, y.shape, y_hat.shape, train_dataset.output_shapes)
            sess.run(train_init_op, feed_dict={train_placeholder : train,
                                               train_l_placeholder : train_l})

            for iteration in range (0, total_iteration_amount, train_batch_size):
                if (iteration % (len(train)) == 0):
                    print( "-"*20 + 'Running Epoch ' + str(int(iteration / len(train))) + "-"*20 )
                elif (iteration % (train_batch_size * 40) == 0):
                    acc, summary = sess.run([acc_raw, train_summary])
                    summary_writer.add_summary(summary, iteration)
                    summary_writer.flush()
                    print('acc_raw at iteration %6d is %.2f' % (iteration, acc))
                else:
                    sess.run(optimiser)

            sess.run(test_init_op, feed_dict={test_placeholder : test,
                                              test_l_placeholder : test_l})

            total_raw_acc = 0;
            total_max_acc = 0;
            total_maj_acc = 0;
            for iteration in range (0, len(test), test_batch_size):
                tmp_raw, tmp_max, tmp_maj = sess.run([acc_raw, acc_max, acc_maj])
                total_raw_acc += tmp_raw
                total_max_acc += tmp_max
                total_maj_acc += tmp_maj
            total_raw_acc /= (len(test) / test_batch_size)
            total_max_acc /= (len(test) / test_batch_size)
            total_maj_acc /= (len(test) / test_batch_size)
            print('Test data acc_raw is %.2f' % (total_raw_acc))
            print('Test data acc_max is %.2f' % (total_max_acc))
            print('Test data acc_maj is %.2f' % (total_maj_acc))




if __name__ == '__main__':
    tf.app.run(main=main)

