import os
import tensorflow as tf
import numpy as np

class Linear(tf.keras.Model):
    def __init__(self):
        super(Linear, self).__init__()

        self.num_classes = 10
        self.batch_size = 25

        self.dense1 = tf.keras.layers.Dense(30, activation='relu', use_bias=True)
        self.dense2 = tf.keras.layers.Dense(self.num_classes, use_bias=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    @tf.function
    def call(self, inputs):
        return self.dense2(self.dense1(inputs))

    @tf.function
    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits))

    @tf.function
    def accuracy(self, logits, labels):
        len = labels.shape[0]
        labels = tf.reshape(labels, [len, 1])
        maxes = tf.cast(tf.argmax(logits, 1), tf.int32)
        correct_predictions = tf.equal(maxes, labels)
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
