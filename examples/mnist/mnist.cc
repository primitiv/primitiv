// Sample code to train/test the MNIST dataset:
//   http://yann.lecun.com/exdb/mnist/
//
// The model consists of a full-connected 2-layer (input/hidden/output)
// perceptron with the softmax cross entropy loss.
//
// Usage:
//   (set include/lib path correctly to use primitiv)
//   $ ./download_data.sh
//   $ g++ -std=c++11 ./mnist.cc -lprimitiv
//   $ ./a.out

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <primitiv/primitiv.h>

#include "utils.h"

using namespace primitiv;
using namespace std;
namespace F = primitiv::functions;
namespace I = primitiv::initializers;
namespace O = primitiv::optimizers;

const unsigned NUM_TRAIN_SAMPLES = 60000;
const unsigned NUM_TEST_SAMPLES = 10000;
const unsigned NUM_INPUT_UNITS = 28 * 28;
const unsigned NUM_HIDDEN_UNITS = 800;
const unsigned NUM_OUTPUT_UNITS = 10;
const unsigned BATCH_SIZE = 200;
const unsigned NUM_TRAIN_BATCHES = NUM_TRAIN_SAMPLES / BATCH_SIZE;
const unsigned NUM_TEST_BATCHES = NUM_TEST_SAMPLES / BATCH_SIZE;
const unsigned MAX_EPOCH = 100;

int main() {
  // Loads data
  vector<float> train_inputs = utils::load_mnist_images(
      "data/train-images-idx3-ubyte", NUM_TRAIN_SAMPLES);
  vector<char> train_labels = utils::load_mnist_labels(
      "data/train-labels-idx1-ubyte", NUM_TRAIN_SAMPLES);
  vector<float> test_inputs = utils::load_mnist_images(
      "data/t10k-images-idx3-ubyte", NUM_TEST_SAMPLES);
  vector<char> test_labels = utils::load_mnist_labels(
      "data/t10k-labels-idx1-ubyte", NUM_TEST_SAMPLES);

  devices::Naive dev;  //devices::CUDA dev(0);
  Device::set_default(dev);
  Graph g;
  Graph::set_default(g);

  // Parameters for the multilayer perceptron.
  Parameter pw1({NUM_HIDDEN_UNITS, NUM_INPUT_UNITS}, I::XavierUniform());
  Parameter pb1({NUM_HIDDEN_UNITS}, I::Constant(0));
  Parameter pw2({NUM_OUTPUT_UNITS, NUM_HIDDEN_UNITS}, I::XavierUniform());
  Parameter pb2({NUM_OUTPUT_UNITS}, I::Constant(0));

  // Optimizer
  O::SGD optimizer(.5);
  optimizer.add(pw1, pb1, pw2, pb2);

  // Helper lambda to construct the predictor network.
  auto make_graph = [&](const vector<float> &inputs, bool train) {
    // Stores input values.
    Node x = F::input<Node>(Shape({NUM_INPUT_UNITS}, BATCH_SIZE), inputs);
    // Calculates the hidden layer.
    Node w1 = F::parameter<Node>(pw1);
    Node b1 = F::parameter<Node>(pb1);
    Node h = F::relu(F::matmul(w1, x) + b1);
    // Dropout
    h = F::dropout(h, .5, train);
    // Calculates the output layer.
    Node w2 = F::parameter<Node>(pw2);
    Node b2 = F::parameter<Node>(pb2);
    return F::matmul(w2, h) + b2;
  };

  // Batch randomizer
  mt19937 rng;
  vector<unsigned> ids(NUM_TRAIN_SAMPLES);
  iota(begin(ids), end(ids), 0);

  for (unsigned epoch = 0; epoch < MAX_EPOCH; ++epoch) {
    // Shuffles sample IDs.
    shuffle(begin(ids), end(ids), rng);

    // Training loop
    for (unsigned batch = 0; batch < NUM_TRAIN_BATCHES; ++batch) {
      // Makes a minibatch for training.
      vector<float> inputs(BATCH_SIZE * NUM_INPUT_UNITS);
      vector<unsigned> labels(BATCH_SIZE);
      for (unsigned i = 0; i < BATCH_SIZE; ++i) {
        const unsigned id = ids[i + batch * BATCH_SIZE];
        copy(&train_inputs[id * NUM_INPUT_UNITS],
             &train_inputs[(id + 1) * NUM_INPUT_UNITS],
             &inputs[i * NUM_INPUT_UNITS]);
        labels[i] = train_labels[id];
      }

      // Constructs the graph.
      g.clear();
      Node y = make_graph(inputs, true);
      Node loss = F::softmax_cross_entropy(y, labels, 0);
      Node avg_loss = F::batch::mean(loss);

      // Dump computation graph at the first time.
      //if (epoch == 0 && batch == 0) cout << g.dump("dot");

      // Implicit forward, backward, and updates parameters.
      optimizer.reset_gradients();
      avg_loss.backward();
      optimizer.update();
    }

    unsigned match = 0;

    // Test loop
    for (unsigned batch = 0; batch < NUM_TEST_BATCHES; ++batch) {
      // Makes a test minibatch.
      vector<float> inputs(BATCH_SIZE * NUM_INPUT_UNITS);
      copy(&test_inputs[batch * BATCH_SIZE * NUM_INPUT_UNITS],
           &test_inputs[(batch + 1) * BATCH_SIZE * NUM_INPUT_UNITS],
           &inputs[0]);

      // Constructs the graph.
      g.clear();
      Node y = make_graph(inputs, false);

      // Gets outputs, argmax, and compares them with the label.
      vector<float> y_val = y.to_vector();
      for (unsigned i = 0; i < BATCH_SIZE; ++i) {
        float maxval = -1e10;
        int argmax = -1;
        for (unsigned j = 0; j < NUM_OUTPUT_UNITS; ++j) {
          float v = y_val[j + i * NUM_OUTPUT_UNITS];
          if (v > maxval) maxval = v, argmax = static_cast<int>(j);
        }
        if (argmax == test_labels[i + batch * BATCH_SIZE]) ++match;
      }
    }

    const float accuracy = 100.0 * match / NUM_TEST_SAMPLES;
    printf("epoch %d: accuracy: %.2f%%\n", epoch, accuracy);
    //pw1.save("mnist-params-w1.param");
    //pb1.save("mnist-params-b1.param");
    //pw2.save("mnist-params-w2.param");
    //pb2.save("mnist-params-b2.param");
    //cout << "epoch " << epoch << ": saved parameters." << endl;
  }

  return 0;
}
