import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape
import numpy as np

class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()

        self.num_classes = 10
        self.batch_size = 100

        model = tf.keras.Sequential()

        model.add(Dense(4 * 4 * 32))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Reshape((4, 4, 32)))

        model.add(Conv2D(16, 5, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Flatten())
        model.add(Dense(15, activation="relu"))
        model.add(Dense(self.num_classes))

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