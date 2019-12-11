import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, MaxPooling2D, AveragePooling2D, Dropout
import numpy as np

class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()

        self.num_classes = 10
        self.batch_size = 100

        model = tf.keras.Sequential()
        weights_init = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)
        model.add(Dense(4 * 4 * 32))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        model.add(Reshape((4, 4, 32)))
        model.add(Conv2D(32, (2, 2), strides=(1,1), activation='relu', kernel_initializer=weights_init))
        # model.add(AveragePooling2D((2,2), strides=(2,2)))
        # discriminator.add(BatchNormalization())
        # discriminator.add(LeakyReLU())

        # model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Conv2D(64, (2, 2), activation='relu', kernel_initializer=weights_init, padding='same'))
        # model.add(AveragePooling2D((2,2), strides=(2,2)))
        # model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Conv2D(64, (2, 2), activation='relu', kernel_initializer=weights_init, padding='same'))
        # model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=weights_init, padding='same'))
        # model.add(AveragePooling2D((2,2), strides=(2,2)))
        #
        # model.add(Conv2D(64, kernel_size=2, strides=2, activation='relu', kernel_initializer=weights_init), padding='same')
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu", use_bias=True))
        # model.add(LeakyReLU())
        # model.add(Dense(16))
        # model.add(Dense(self.num_classes, activation='softmax'))
        model.add(Dense(self.num_classes, use_bias=True))

        # model.add(Dense(4 * 4 * 32))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU(0.2))
        # model.add(Reshape((4, 4, 32)))
        #
        # # model.add(Conv2D(32, 5, strides=2, padding="same", kernel_initializer=weights_init))
        # # model.add(BatchNormalization())
        # # model.add(LeakyReLU())
        #
        # model.add(Conv2D(16, 5, strides=2, padding="same", kernel_initializer=weights_init))
        # model.add(BatchNormalization())
        # model.add(LeakyReLU())
        #
        # model.add(Flatten())
        # model.add(Dense(15, activation="relu"))
        # # model.add(Dense(8, activation="relu"))
        # model.add(Dense(units=self.num_classes))

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
