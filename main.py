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
            # print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
    print("Accuracy")
    logits = model.call(test_inputs)
    return model.accuracy(logits, test_labels)

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"LINEAR","CNN", "RNN"}:
        print("USAGE: python main.py <Model Type>")
        print("<Model Type>: [LINEAR/CNN/RNN]")
        exit()

    print("Running preprocessing...")
    # train_inputs, train_labels, test_inputs, test_labels = get_data("data/genres.tar")
    # train_inputs, train_labels, test_inputs, test_labels = get_data("genres.gz")
    if sys.argv[1] in {"RNN"}:
        train_inputs, train_labels, test_inputs, test_labels = get_rnn_data("data/genres.tar")
    else:
        train_inputs, train_labels, test_inputs, test_labels = get_data("data/genres.tar")

    print("Preprocessing complete.")

    if sys.argv[1] == "LINEAR":
        model = Linear()
        num_epochs = 100
    elif sys.argv[1] == "CNN":
        model = CNN()
        num_epochs = 1
    elif sys.argv[1] == "RNN":
        model = RNN()
        num_epochs = 1


    for _ in range(num_epochs):
        train(model, train_inputs, train_labels)

    # history = model.fit(train_inputs, train_labels, validation_split = 0.2, epochs=num_epochs, batch_size=model.batch_size, verbose=1)
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    #
    # # Plot training & validation loss values
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    #
    # from keras.utils import plot_model
    # plot_model(model, to_file="model.png")
    print(test(model, test_inputs, test_labels))


if __name__ == '__main__':
   main()
