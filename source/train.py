from __future__ import print_function

import os
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from source import createdb

PATCH = [33, 33]
NUM_CHANNELS = [4, 9]
NUM_LABEL = 5
KERNEL = [3, 3]
FILTERS = [64, 128, 256]
NUM_EPOCHS = 20
BATCH_SIZE = 128
EVAL_FREQUENCY = 100

WRITE_PATH = createdb.WRITE_PATH


def get_conv_weight_bias(name, filtercnt, shape):
    weights = tf.get_variable(name=name + "w", shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              dtype=tf.float32)
    biases = tf.Variable(tf.constant(0.1, shape=[filtercnt], dtype=tf.float32), name=name + "b")
    return weights, biases


def get_fc_weight_bias(name, inputcnt, filtercnt):
    weights = tf.get_variable(name=name + "w", shape=([inputcnt, filtercnt]),
                              initializer=tf.contrib.layers.xavier_initializer_conv2d(), dtype=tf.float32)
    biases = tf.Variable(tf.constant(0.1, shape=[filtercnt], dtype=tf.float32), name=name + "b")
    return weights, biases


def get_conv_layer(name, data, weight, bias):
    conv = tf.nn.conv2d(input=data, filter=weight, strides=[1, 1, 1, 1], padding="SAME", name=name + "l")
    return get_leaky_relu(tf.nn.bias_add(conv, bias))


def get_leaky_relu(data):
    return tf.nn.relu(data)


def get_maxpool_layer(name, data):
    return tf.nn.max_pool(value=data, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name=name)


def get_fc_layer(data, weight, bias, dropout):
    shp = data.get_shape().as_list()
    newshp = 1
    for i in range(1, len(shp)):
        newshp *= shp[i]
    hidden = tf.nn.bias_add(tf.matmul(tf.reshape(data, [shp[0], newshp]), weight), bias)
    hidden = get_leaky_relu(hidden)
    hidden = tf.nn.dropout(hidden, dropout)
    return hidden


def get_output_layer(data, weight, bias, label):
    shp = data.get_shape().as_list()
    newshp = 1
    for i in range(1, len(shp)):
        newshp *= shp[i]
    hidden = tf.reshape(data, [shp[0], newshp])
    hidden = tf.nn.bias_add(tf.matmul(hidden, weight), bias)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(hidden, label)), tf.nn.softmax(hidden)


def model(train_data_node, train_labels_node):
    c1w, c1b = get_conv_weight_bias("c1", FILTERS[0], [KERNEL[0], KERNEL[1], NUM_CHANNELS[0], FILTERS[0]])
    c2w, c2b = get_conv_weight_bias("c2", FILTERS[0], [KERNEL[0], KERNEL[1], FILTERS[0], FILTERS[0]])
    c3w, c3b = get_conv_weight_bias("c3", FILTERS[1], [KERNEL[0], KERNEL[1], FILTERS[0], FILTERS[1]])
    c4w, c4b = get_conv_weight_bias("c4", FILTERS[1], [KERNEL[0], KERNEL[1], FILTERS[1], FILTERS[1]])
    c5w, c5b = get_conv_weight_bias("c5", FILTERS[1], [KERNEL[0], KERNEL[1], FILTERS[1], FILTERS[1]])
    c6w, c6b = get_conv_weight_bias("c6", FILTERS[1], [KERNEL[0], KERNEL[1], FILTERS[1], FILTERS[1]])

    f1w, f1b = get_fc_weight_bias("f1", 7 * 7 * 128, FILTERS[2])
    f2w, f2b = get_fc_weight_bias("f2", FILTERS[2], FILTERS[2])
    f3w, f3b = get_fc_weight_bias("f3", FILTERS[2], NUM_LABEL)

    hidden1 = get_conv_layer("c1", train_data_node, c1w, c1b)
    hidden2 = get_conv_layer("c2", hidden1, c2w, c2b)
    hidden3 = get_conv_layer("c3", hidden2, c3w, c3b)
    hidden4 = get_maxpool_layer("p1", hidden3)
    hidden5 = get_conv_layer("c4", hidden4, c4w, c4b)
    hidden6 = get_conv_layer("c5", hidden5, c5w, c5b)
    hidden7 = get_conv_layer("c6", hidden6, c6w, c6b)
    hidden8 = get_maxpool_layer("p2", hidden7)
    hidden9 = get_fc_layer(hidden8, f1w, f1b, 0.5)
    hidden10 = get_fc_layer(hidden9, f2w, f2b, 0.5)
    loss, train_prediction = get_output_layer(hidden10, f3w, f3b, train_labels_node)
    return loss, train_prediction


def train(dataset):
    train_labels = np.memmap(os.path.join(WRITE_PATH, dataset + "_train.lbl"), dtype=np.int8, mode="r")
    train_data = np.memmap(os.path.join(WRITE_PATH, dataset + "_train.dat"), dtype=np.float32,
                           shape=(train_labels.shape[0], PATCH[0], PATCH[1], NUM_CHANNELS[0]), mode="r")
    train_size = train_labels.shape[0]
    train_data_node = tf.placeholder(tf.float32,
                                     shape=(BATCH_SIZE, PATCH[0], PATCH[1], NUM_CHANNELS[0]))
    train_labels_node = tf.placeholder(tf.int64, shape=BATCH_SIZE)

    loss, train_prediction = model(train_data_node, train_labels_node)

    batch = tf.Variable(0, dtype=tf.float32)
    learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=batch * BATCH_SIZE,
                                               decay_steps=train_size, staircase=True, decay_rate=0.7943282347242815)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
    predict = tf.to_double(100) * (
        tf.to_double(1) - tf.reduce_mean(tf.to_double(tf.nn.in_top_k(train_prediction, train_labels_node, 1))))

    start_time = time.time()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print("Variable Initialized")
        tf.scalar_summary("error", predict)
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(WRITE_PATH + "summary", sess.graph)
        cnttt = 0
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=2, max_to_keep=30)
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print(total_parameters)
        for step in range(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:offset + BATCH_SIZE]
            batch_labels = train_labels[offset:offset + BATCH_SIZE]
            feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}
            _, l, lr, predictions, summary_out = sess.run(
                [optimizer, loss, learning_rate, predict, summary_op],
                feed_dict=feed_dict)
            if step % EVAL_FREQUENCY == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                summary_writer.add_summary(summary_out, global_step=step)
                print(str(datetime.now()))
                print('Step %d (epoch %.2f), %d s' %
                      (step, float(step) * BATCH_SIZE / train_size, elapsed_time))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % predictions)
                sys.stdout.flush()
            if cnttt != (step * BATCH_SIZE) / train_size:
                print("Saved in path", saver.save(sess, WRITE_PATH + "savedmodel/" + str(cnttt) + ".ckpt"))
            cnttt = (step * BATCH_SIZE) / train_size
        print("Saved in path", saver.save(sess, WRITE_PATH + "savedmodel/savedmodel_final.ckpt"))
    tf.reset_default_graph()


def main(argv=None):
    train("h")


if __name__ == '__main__':
    tf.app.run()
