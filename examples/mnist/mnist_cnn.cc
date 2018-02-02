// Sample code to train/test the MNIST dataset:
//   http://yann.lecun.com/exdb/mnist/
//
// The model consists of 2 CNNs and 2 FC layers:
// input                        {28, 28, 1} x B
//   -> conv2d + relu           {28, 28, C1} x B
//   -> max_pool2d              {14, 14, C1} x B
//   -> conv2d + relu           {14, 14, C2} x B
//   -> max_pool2d              {7, 7, C2} x B
//   -> flatten + dropout       {7 * 7 * C2} x B
//   -> affine + relu + dropout {H} x B
//   -> affine                  {10} x B
// ( -> softmax_cross_entropy ) {1} x B
// ( -> batch_mean            ) {1}
//
// Usage:
//   (set include/lib path correctly to use primitiv)
//   $ ./download_data.sh
//   $ g++ -std=c++11 ./mnist_cnn.cc -lprimitiv
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
const unsigned BATCH_SIZE = 200;
const unsigned NUM_TRAIN_BATCHES = NUM_TRAIN_SAMPLES / BATCH_SIZE;
const unsigned NUM_TEST_BATCHES = NUM_TEST_SAMPLES / BATCH_SIZE;
const unsigned MAX_EPOCH = 100;

const unsigned IMAGE_HEIGHT = 28;
const unsigned IMAGE_WIDTH = 28;

const unsigned KERNEL_SIZE1 = 5;  // should be an odd number
const unsigned KERNEL_SIZE2 = 5;  // ditto
const unsigned NUM_CHANNELS1 = 8;
const unsigned NUM_CHANNELS2 = 16;
const unsigned PADDING1 = KERNEL_SIZE1 / 2;
const unsigned PADDING2 = KERNEL_SIZE2 / 2;

const unsigned NUM_INPUT_UNITS
  = (IMAGE_HEIGHT / 4) * (IMAGE_WIDTH / 4) * NUM_CHANNELS2;
const unsigned NUM_HIDDEN_UNITS = 256;
const unsigned NUM_OUTPUT_UNITS = 10;

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

  devices::CUDA dev(0);
  Device::set_default(dev);
  Graph g;
  Graph::set_default(g);

  // Parameters of CNNs
  // Shape: {kernel_height, kernel_width, in_channels, out_channels}
  Parameter pw_cnn1(
      {KERNEL_SIZE1, KERNEL_SIZE1, 1, NUM_CHANNELS1},
      I::XavierUniformConv2D());
  Parameter pw_cnn2(
      {KERNEL_SIZE2, KERNEL_SIZE2, NUM_CHANNELS1, NUM_CHANNELS2},
      I::XavierUniformConv2D());

  // Parameters of FC layers
  Parameter pw_fc1({NUM_HIDDEN_UNITS, NUM_INPUT_UNITS},  I::XavierUniform());
  Parameter pw_fc2({NUM_OUTPUT_UNITS, NUM_HIDDEN_UNITS}, I::XavierUniform());
  Parameter pb_fc1({NUM_HIDDEN_UNITS}, I::Constant(0));
  Parameter pb_fc2({NUM_OUTPUT_UNITS}, I::Constant(0));

  // Optimizer
  O::SGD optimizer(.1);
  optimizer.add(pw_cnn1, pw_cnn2, pw_fc1, pw_fc2, pb_fc1, pb_fc2);

  // Helper lambda to construct the predictor network.
  auto make_graph = [&](const vector<float> &inputs, bool train) {
    // Input and parameters.
    const Node x = F::input<Node>(
        Shape({IMAGE_HEIGHT, IMAGE_WIDTH}, BATCH_SIZE), inputs);
    const Node w_cnn1 = F::parameter<Node>(pw_cnn1);
    const Node w_cnn2 = F::parameter<Node>(pw_cnn2);
    const Node w_fc1 = F::parameter<Node>(pw_fc1);
    const Node w_fc2 = F::parameter<Node>(pw_fc2);
    const Node b_fc1 = F::parameter<Node>(pb_fc1);
    const Node b_fc2 = F::parameter<Node>(pb_fc2);
    // CNNs
    const Node h_cnn1 = F::relu(
        F::conv2d(x, w_cnn1, PADDING1, PADDING1, 1, 1, 1, 1));
    const Node h_pool1 = F::max_pool2d(h_cnn1, 2, 2, 0, 0, 2, 2);
    const Node h_cnn2 = F::relu(
        F::conv2d(h_pool1, w_cnn2, PADDING2, PADDING2, 1, 1, 1, 1));
    const Node h_pool2 = F::max_pool2d(h_cnn2, 2, 2, 0, 0, 2, 2);
    // FC layers
    const Node x_fc = F::dropout(F::flatten(h_pool2), .5, train);
    const Node h_fc = F::dropout(
        F::relu(F::matmul(w_fc1, x_fc) + b_fc1), .5, train);
    return F::matmul(w_fc2, h_fc) + b_fc2;
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
      vector<float> inputs(BATCH_SIZE * IMAGE_HEIGHT * IMAGE_WIDTH);
      vector<unsigned> labels(BATCH_SIZE);
      for (unsigned i = 0; i < BATCH_SIZE; ++i) {
        const unsigned id = ids[i + batch * BATCH_SIZE];
        copy(&train_inputs[id * IMAGE_HEIGHT * IMAGE_WIDTH],
             &train_inputs[(id + 1) * IMAGE_HEIGHT * IMAGE_WIDTH],
             &inputs[i * IMAGE_HEIGHT * IMAGE_WIDTH]);
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
      vector<float> inputs(BATCH_SIZE * IMAGE_HEIGHT * IMAGE_WIDTH);
      copy(&test_inputs[batch * BATCH_SIZE * IMAGE_HEIGHT * IMAGE_WIDTH],
           &test_inputs[(batch + 1) * BATCH_SIZE * IMAGE_HEIGHT * IMAGE_WIDTH],
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
  }

  return 0;
}
