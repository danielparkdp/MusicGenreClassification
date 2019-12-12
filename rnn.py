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
<<<<<<< HEAD
        # model.add(Dense(4 * 4 * 32))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(0.2))
        # model.add(Reshape((16, 32)))
        # model.add(GRU(90, use_bias=True, dropout=0.5, recurrent_dropout=0.2))
        # model.add(Dense(60, kernel_regularizer=l2(.01), use_bias=True))
        # model.add(LeakyReLU(0.5))
        # model.add(Dense(self.num_classes, activation='softmax', use_bias=True))
        # model.add(Dense(100, activation="relu"))
        # model.add(Reshape((5, 20)))
        # model.add(GRU(20, use_bias=True, dropout=0.5, recurrent_dropout=0.1))
        # model.add(Dense(60, kernel_regularizer=l2(.01), use_bias=True))
        # model.add(Dense(self.num_classes, activation='softmax', use_bias=True))

        model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True))
        model.add(LSTM(units=32,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
        model.add(Dense(units=self.num_classes, activation="softmax"))
=======
        #model.add(Dense(4 * 4 * 32))
        #model.add(BatchNormalization())
        #model.add(LeakyReLU(0.2))
        #model.add(Reshape((16, 32)))
        #model.add(GRU(90, use_bias=True, dropout=0.5, recurrent_dropout=0.2))
        #model.add(Dense(60, kernel_regularizer=l2(.01), use_bias=True))
        #model.add(LeakyReLU(0.5))
        #model.add(Dense(self.num_classes, activation='softmax', use_bias=True))
        # model.add(Dense(100, activation="relu"))
        #model.add(Reshape((5, 20)))
        #model.add(GRU(20, use_bias=True, dropout=0.5, recurrent_dropout=0.1))
        #model.add(Dense(60, kernel_regularizer=l2(.01), use_bias=True))
        #model.add(Dense(self.num_classes, activation='softmax', use_bias=True))

        #model.add(LSTM(units=128, dropout=0.5, recurrent_dropout=0.10, return_sequences=True))
        model.add(LSTM(units=128,  dropout=0.5, recurrent_dropout=0.10, return_sequences=False))
        model.add(Dense(150, activation="relu", use_bias=True))
        model.add(Dropout(0.5))
        model.add(Dense(units=self.num_classes, activation="softmax", use_bias=True))

        # model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True))
        # model.add(LSTM(units=32,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
        # model.add(Dense(units=self.num_classes, activation="softmax"))
>>>>>>> 67217ac1bbc5ebd5785d5cac967ece324cbffb95


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
