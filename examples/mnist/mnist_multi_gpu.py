#!/usr/bin/env python3

from primitiv import Graph
from primitiv import Parameter

from primitiv import devices as D
from primitiv import initializers as I
from primitiv import trainers as T
from primitiv import operators as F

import random
import sys
import numpy as np


NUM_TRAIN_SAMPLES = 60000
NUM_TEST_SAMPLES = 10000
NUM_INPUT_UNITS = 28 * 28
NUM_HIDDEN_UNITS = 800
NUM_OUTPUT_UNITS = 10
BATCH_SIZE = 50
NUM_TRAIN_BATCHES = NUM_TRAIN_SAMPLES // BATCH_SIZE
NUM_TEST_BATCHES = NUM_TEST_SAMPLES // BATCH_SIZE
MAX_EPOCH = 100


def load_images(filename, n):
    with open(filename, "rb") as ifs:
        ifs.seek(16)  # header
        return (np.fromfile(ifs, dtype=np.uint8, count=n*NUM_INPUT_UNITS) / 255) \
            .astype(np.float32) \
            .reshape((n, NUM_INPUT_UNITS))


def load_labels(filename, n):
    with open(filename, "rb") as ifs:
        ifs.seek(8)  # header
        return np.fromfile(ifs, dtype=np.uint8, count=n) \
            .astype(np.uint32)


def main():
    # Loads data
    train_inputs = load_images("data/train-images-idx3-ubyte", NUM_TRAIN_SAMPLES)
    train_labels = load_labels("data/train-labels-idx1-ubyte", NUM_TRAIN_SAMPLES)
    test_inputs = load_images("data/t10k-images-idx3-ubyte", NUM_TEST_SAMPLES)
    test_labels = load_labels("data/t10k-labels-idx1-ubyte", NUM_TEST_SAMPLES)

    # Initializes 2 device objects which manage different GPUs.
    dev0 = D.CUDA(0)
    dev1 = D.CUDA(1)

    # Parameters on GPU 0.
    pw1 = Parameter([NUM_HIDDEN_UNITS, NUM_INPUT_UNITS], I.XavierUniform(), dev0)
    pb1 = Parameter([NUM_HIDDEN_UNITS], I.Constant(0), dev0)

    # Parameters on GPU 1.
    pw2 = Parameter([NUM_OUTPUT_UNITS, NUM_HIDDEN_UNITS], I.XavierUniform(), dev1)
    pb2 = Parameter([NUM_OUTPUT_UNITS], I.Constant(0), dev1)

    trainer = T.SGD(.1)
    trainer.add_parameter(pw1)
    trainer.add_parameter(pb1)
    trainer.add_parameter(pw2)
    trainer.add_parameter(pb2)

    def make_graph(inputs):
        # We first store input values explicitly on GPU 0.
        x = F.input(inputs, device=dev0)
        w1 = F.parameter(pw1)
        b1 = F.parameter(pb1)
        w2 = F.parameter(pw2)
        b2 = F.parameter(pb2)
        # The hidden layer is calculated and implicitly stored on GPU 0.
        h_on_gpu0 = F.relu(w1 @ x + b1)
        # `copy()` transfers the hiddne layer to GPU 1.
        h_on_gpu1 = F.copy(h_on_gpu0, dev1)
        # The output layer is calculated and implicitly stored on GPU 1.
        return w2 @ h_on_gpu1 + b2

    ids = list(range(NUM_TRAIN_SAMPLES))

    g = Graph()
    Graph.set_default(g)

    for epoch in range(MAX_EPOCH):
        random.shuffle(ids)

        # Training loop
        for batch in range(NUM_TRAIN_BATCHES):
            print("\rTraining... %d / %d" % (batch + 1, NUM_TRAIN_BATCHES), end="")
            inputs = [train_inputs[ids[batch * BATCH_SIZE + i]] for i in range(BATCH_SIZE)]
            labels = [train_labels[ids[batch * BATCH_SIZE + i]] for i in range(BATCH_SIZE)]

            g.clear()

            y = make_graph(inputs)
            loss = F.softmax_cross_entropy(y, labels, 0)
            avg_loss = F.batch.mean(loss)

            trainer.reset_gradients()
            avg_loss.backward()
            trainer.update()

        print()

        match = 0

        # Test loop
        for batch in range(NUM_TEST_BATCHES):
            print("\rTesting... %d / %d" % (batch + 1, NUM_TEST_BATCHES), end="")
            inputs = [test_inputs[batch * BATCH_SIZE + i] for i in range(BATCH_SIZE)]

            g.clear()

            y = make_graph(inputs)
            y_val = y.to_list()
            for i in range(BATCH_SIZE):
                maxval = -1e10
                argmax = -1
                for j in range(NUM_OUTPUT_UNITS):
                    v = y_val[j + i * NUM_OUTPUT_UNITS]
                    if (v > maxval):
                        maxval = v
                        argmax = j
                if argmax == test_labels[i + batch * BATCH_SIZE]:
                    match += 1

        accuracy = 100.0 * match / NUM_TEST_SAMPLES
        print("\nepoch %d: accuracy: %.2f%%\n" % (epoch, accuracy))


if __name__ == "__main__":
    main()
