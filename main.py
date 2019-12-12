import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
import sys
import matplotlib.pyplot as plt

from linear import Linear
from cnn import CNN
from rnn import RNN

def train(model, train_inputs, train_labels):
    for batch_num in range(0, len(train_inputs), model.batch_size):
        with tf.GradientTape() as tape:
            logits = model.call(train_inputs[batch_num : batch_num + model.batch_size])
            loss = model.loss(logits, train_labels[batch_num : batch_num + model.batch_size])
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
    logits = model.call(test_inputs)
    return model.accuracy(logits, test_labels)

def main():
    if sys.argv[1] == "LINEAR":
        model = Linear()
        num_epochs = 100
    elif sys.argv[1] == "CNN":
        model = CNN()
        num_epochs = 100
    elif sys.argv[1] == "RNN":
        model = RNN()
        num_epochs = 100
    if len(sys.argv) != 2 or sys.argv[1] not in {"LINEAR","CNN", "RNN"}:
        print("USAGE: python main.py <Model Type>")
        print("<Model Type>: [LINEAR/CNN/RNN]")
        exit()

    print("Running preprocessing...")
    if sys.argv[1] == "RNN":
        train_inputs, train_labels, test_inputs, test_labels = get_rnn_data("data/genres.tar")
    else:
        train_inputs, train_labels, test_inputs, test_labels = get_data("data/genres.tar")
    print("Preprocessing completed.")
    
    if sys.argv[1] == "RNN":
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=256))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(0.2))
        model.add(tf.keras.layers.Reshape((16, 16)))
        model.add(tf.keras.layers.LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True))
        model.add(tf.keras.layers.LSTM(units=32,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
        model.add(tf.keras.layers.Dense(units=10, activation="softmax"))

        num_epochs = 100
        opt = tf.keras.optimizers.Adam(lr=.01)
        model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history=model.fit(train_inputs, train_labels, epochs=num_epochs, batch_size=100)
        model.summary()
        
        test_loss, test_acc = model.evaluate(test_inputs, test_labels)
        print('test acc: ', test_acc)
    else:
        for _ in range(num_epochs):
            train(model, train_inputs, train_labels)
        print(test(model, test_inputs, test_labels))

if __name__ == '__main__':
   main()
