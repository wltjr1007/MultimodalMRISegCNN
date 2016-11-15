from __future__ import print_function
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf
from medpy.io import load
from medpy.io import save

from source import createdb

PATCH = [33, 33]
NUM_CHANNELS = 4
NUM_LABEL = 5
KERNEL = [3, 3]
FILTERS = [64, 128, 256]
SHAPE = (4, 240, 240, 155)
BATCH_SIZE = 155
BATCH_MUL = 1
VAL_SIZE = 25
ORIG_READ_PATH = createdb.ORIG_READ_PATH
WRITE_PATH = createdb.WRITE_PATH

ALLMEAN, ALLSTD = np.load(os.path.join(WRITE_PATH, "h_zmuv.npy"))


def get_conv_weight_bias(name, filtercnt, shape):
    weights = tf.get_variable(name=name + "w", shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              dtype=tf.float32, trainable=False)
    biases = tf.Variable(tf.constant(0.1, shape=[filtercnt], dtype=tf.float32), name=name + "b", trainable=False)
    return weights, biases


def get_fc_weight_bias(name, inputcnt, filtercnt):
    weights = tf.get_variable(name=name + "w", shape=([inputcnt, filtercnt]),
                              initializer=tf.contrib.layers.xavier_initializer_conv2d(), dtype=tf.float32,
                              trainable=False)
    biases = tf.Variable(tf.constant(0.1, shape=[filtercnt], dtype=tf.float32), name=name + "b", trainable=False)
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


def get_output_layer(data, weight, bias):
    shp = data.get_shape().as_list()
    newshp = 1
    for i in range(1, len(shp)):
        newshp *= shp[i]
    hidden = tf.reshape(data, [shp[0], newshp])
    hidden = tf.nn.bias_add(tf.matmul(hidden, weight), bias)
    return tf.nn.softmax(hidden)


def test_model(test_data_node):
    c1w, c1b = get_conv_weight_bias("c1", FILTERS[0], [KERNEL[0], KERNEL[1], NUM_CHANNELS, FILTERS[0]])
    c2w, c2b = get_conv_weight_bias("c2", FILTERS[0], [KERNEL[0], KERNEL[1], FILTERS[0], FILTERS[0]])
    c3w, c3b = get_conv_weight_bias("c3", FILTERS[1], [KERNEL[0], KERNEL[1], FILTERS[0], FILTERS[1]])
    c4w, c4b = get_conv_weight_bias("c4", FILTERS[1], [KERNEL[0], KERNEL[1], FILTERS[1], FILTERS[1]])
    c5w, c5b = get_conv_weight_bias("c5", FILTERS[1], [KERNEL[0], KERNEL[1], FILTERS[1], FILTERS[1]])
    c6w, c6b = get_conv_weight_bias("c6", FILTERS[1], [KERNEL[0], KERNEL[1], FILTERS[1], FILTERS[1]])

    f1w, f1b = get_fc_weight_bias("f1", 7 * 7 * 128, FILTERS[2])
    f2w, f2b = get_fc_weight_bias("f2", FILTERS[2], FILTERS[2])
    f3w, f3b = get_fc_weight_bias("f3", FILTERS[2], NUM_LABEL)

    hidden1 = get_conv_layer("c1", test_data_node, c1w, c1b)
    hidden2 = get_conv_layer("c2", hidden1, c2w, c2b)
    hidden3 = get_conv_layer("c3", hidden2, c3w, c3b)
    hidden4 = get_maxpool_layer("p1", hidden3)
    hidden5 = get_conv_layer("c4", hidden4, c4w, c4b)
    hidden6 = get_conv_layer("c5", hidden5, c5w, c5b)
    hidden7 = get_conv_layer("c6", hidden6, c6w, c6b)
    hidden8 = get_maxpool_layer("p2", hidden7)
    hidden9 = get_fc_layer(hidden8, f1w, f1b, 0.5)
    hidden10 = get_fc_layer(hidden9, f2w, f2b, 0.5)
    test_prediction = get_output_layer(hidden10, f3w, f3b)
    return test_prediction


def test(data_name_list):
    test_data = np.memmap(os.path.join(WRITE_PATH, "test_orig.dat"), dtype=np.float32, mode="r",
                          shape=(110, SHAPE[1], SHAPE[2], SHAPE[3], SHAPE[0]))
    val_data = np.memmap(os.path.join(WRITE_PATH, "h_orig.dat"), shape=(220, SHAPE[0], SHAPE[1], SHAPE[2], SHAPE[3]),
                         dtype=np.float32, mode="r")
    test_size = test_data.shape[0]
    test_data_node = tf.placeholder(tf.float32,
                                    shape=(BATCH_SIZE * BATCH_MUL, PATCH[0], PATCH[1], NUM_CHANNELS))
    test_prediction = test_model(test_data_node)
    imghdr = load(ORIG_READ_PATH + "h.1.VSD.Brain.XX.O.MR_Flair.54512.nii")[1]
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        saver.restore(sess, WRITE_PATH + "savedmodel/savedmodel_final.ckpt")
        print("Variable Initialized. Start Testing!")
        for i in range(test_size + VAL_SIZE):
            test_time = time.time()
            test_result = np.zeros(dtype=np.uint8, shape=(SHAPE[1], SHAPE[2], SHAPE[3]))
            for j in range(SHAPE[1]):
                for k in range(0, SHAPE[2], BATCH_MUL):
                    if i < VAL_SIZE:
                        batch_data, is_background = get_val_data(val_data, i, j, k)
                    else:
                        batch_data, is_background = get_test_data(test_data, i - VAL_SIZE, j, k)
                    if is_background:
                        continue
                    feed_dict = {test_data_node: batch_data}
                    test_result[j, k] = np.argmax(sess.run(test_prediction, feed_dict=feed_dict),
                                                  1)
            if i < VAL_SIZE:
                test_result[np.where(val_data[i, 0] == 0)] = 0
                save(test_result, WRITE_PATH + "VSD.h." + str(i) + "." + data_name_list[0][i, 3] + ".nii", imghdr)
            else:
                test_result[np.where(test_data[i - VAL_SIZE, 0] == 0)] = 0
                save(test_result,
                     WRITE_PATH + "VSD.t." + str(i - VAL_SIZE) + "." + data_name_list[2][i - VAL_SIZE, 3] + ".nii",
                     imghdr)

            print("TEST %d/%d, Time elapsed: %d" % (i - VAL_SIZE, test_size, time.time() - test_time))


def get_test_data(datas, patient, width, height):
    w_h = [width - PATCH[0] / 2, width + PATCH[0] / 2 + 1, height - PATCH[1] / 2,
           height + PATCH[1] / 2 + 1]
    pad = [min(PATCH[0] / 2, max(0, -w_h[0])), min(PATCH[0] / 2, max(0, w_h[1] - SHAPE[1])),
           min(PATCH[1] / 2, max(0, -w_h[2])), min(PATCH[1] / 2, max(0, w_h[3] - SHAPE[2]))]
    data = datas[patient, max(0, w_h[0]):min(SHAPE[1], w_h[1]), max(0, w_h[2]):min(SHAPE[2], w_h[3]), :, :]
    result = np.zeros((155, 33, 33, 4), dtype=np.float32)
    if np.all(data == 0):
        return result, True
    if data.shape != (33, 33, 155, 4):
        data = np.pad(data, ((pad[0:2]), (pad[2:4]), (0, 0), (0, 0)), "constant")
    for i in range(SHAPE[0]):
        np.true_divide(np.subtract(np.rollaxis(data[..., i], 2, 0), ALLMEAN[i]), ALLSTD[i], result[..., i])
    return result, False


def get_val_data(datas, patient, width, height):
    w_h = [width - PATCH[0] / 2, width + PATCH[0] / 2 + 1, height - PATCH[1] / 2,
           height + PATCH[1] / 2 + 1]
    pad = [min(PATCH[0] / 2, max(0, -w_h[0])), min(PATCH[0] / 2, max(0, w_h[1] - SHAPE[1])),
           min(PATCH[1] / 2, max(0, -w_h[2])), min(PATCH[1] / 2, max(0, w_h[3] - SHAPE[2]))]
    data = datas[patient, :, max(0, w_h[0]):min(SHAPE[1], w_h[1]), max(0, w_h[2]):min(SHAPE[2], w_h[3]), :]
    result = np.zeros((155, 33, 33, 4), dtype=np.float32)
    if np.all(data == 0):
        return result, True
    if data.shape != (4, 33, 33, 155):
        data = np.pad(data, ((0, 0), (pad[0:2]), (pad[2:4]), (0, 0)), "constant")
    for i in range(SHAPE[0]):
        np.true_divide(np.subtract(np.rollaxis(data[i], 2, 0), ALLMEAN[i]), ALLSTD[i], result[..., i])
    return result, False


def get_img_name_list(path):
    mods = {"T1": 0, "T2": 1, "T1c": 2, "Flair": 3, "OT": 4}
    name_list = os.listdir(path)
    result_list_l = {}
    result_list_h = {}
    result_list_t = {}
    for name in name_list:
        temp = name.replace(".nii", "").split(".")
        dataset = temp[0]
        cnt = int(temp[1]) - 1
        mod = mods[temp[-2].split("_")[-1]]
        dbcnt = temp[-1]
        if dataset == "l":
            result_list_l[cnt, mod] = dbcnt
        elif dataset == "h":
            result_list_h[cnt, mod] = dbcnt
        elif dataset == "t":
            result_list_t[cnt, mod] = dbcnt
    return np.array([result_list_h, result_list_l, result_list_t])


def main(argv=None):
    data_name_list = get_img_name_list(ORIG_READ_PATH)
    test(data_name_list)


if __name__ == '__main__':
    tf.app.run()
