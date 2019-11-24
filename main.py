import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
import sys

from linear import Linear

def train(model, train_inputs, train_labels):
    for batch_num in range(0, len(train_inputs), model.batch_size):
        with tf.GradientTape() as tape:
            logits = model.call(train_inputs[batch_num : batch_num + model.batch_size])
            loss = model.loss(logits, train_labels[batch_num : batch_num + model.batch_size])
            print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
	logits = model.call(test_inputs)
	return model.accuracy(logits, test_labels)

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"LINEAR","CNN", "RNN"}:
        print("USAGE: python main.py <Model Type>")
        print("<Model Type>: [LINEAR/CNN/RNN]")
        exit()

    print("Running preprocessing...")
    train_inputs, train_labels, test_inputs, test_labels = get_data("data/genres.tar")
    print("Preprocessing complete.")

    # Model arguments
	# TODO: Set model arguments here
    if sys.argv[1] == "LINEAR":
        model = Linear()
    elif sys.argv[1] == "CNN":
        model = None
    elif sys.argv[1] == "RNN":
        model = None
        
	# TODO: train and test
    train(model, train_inputs, train_labels)
    print(test(model, test_inputs, test_labels))


if __name__ == '__main__':
   main()
