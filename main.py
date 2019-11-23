import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
import sys


def train(model, train_inputs, train_labels):
	pass

def test(model, test_inputs, test_labels):
	pass

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"LINEAR","CNN", "RNN"}:
        print("USAGE: python main.py <Model Type>")
        print("<Model Type>: [LINEAR/CNN/RNN]")
        exit()

    print("Running preprocessing...")
	# TODO: Get data here
    inputs, labels = get_data("data/genres.tar")
    print("Preprocessing complete.")

    # Model arguments
	# TODO: Set model arguments here
    if sys.argv[1] == "LINEAR":
        model = None
    elif sys.argv[1] == "CNN":
        model = None
    elif sys.argv[1] == "RNN":
        model = None
        
	# TODO: train and test


if __name__ == '__main__':
   main()
