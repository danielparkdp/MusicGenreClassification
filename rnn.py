import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape, GRU, InputLayer, LSTM, Dropout
import numpy as np

class RNN(tf.keras.Model):
    def __init__(self):
        super(RNN, self).__init__()

        self.num_classes = 10
        self.batch_size = 200

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=256))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(0.2))
        model.add(tf.keras.layers.Reshape((16, 16)))
        model.add(tf.keras.layers.LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True))
        model.add(tf.keras.layers.LSTM(units=32,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
        model.add(tf.keras.layers.Dense(units=10, activation="softmax"))

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
