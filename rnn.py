import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, GRU
import numpy as np

class RNN(tf.keras.Model):
    def __init__(self):
        super(RNN, self).__init__()

        self.num_classes = 10
        self.batch_size = 100

        model = tf.keras.Sequential()
        model.add(Dense(4 * 4 * 32))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Reshape((16, 32)))
        model.add(GRU(90, use_bias=True, dropout=0.5, recurrent_dropout=0.2))
        model.add(Dense(60, kernel_regularizer=l2(.01), use_bias=True))
        model.add(LeakyReLU(0.5))
        model.add(Dense(self.num_classes, activation='softmax', use_bias=True))

        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def call(self, inputs):
        return self.model(inputs)

    @tf.function
    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))

    @tf.function
    def accuracy(self, logits, labels):
        maxes = tf.cast(tf.argmax(logits, 1), tf.int32)
        correct = tf.equal(maxes, labels)
        return tf.reduce_mean(tf.cast(correct, tf.float32))
