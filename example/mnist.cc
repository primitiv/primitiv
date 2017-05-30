// Sample code to train/test the MNIST dataset:
//   http://yann.lecun.com/exdb/mnist/
//
// The model consists of a full-connected 2-layer (input/hidden/output)
// perceptron with 300 hidden units, and is trained using the squared loss.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <primitiv/primitiv.h>
#include <primitiv/primitiv_cuda.h>

using primitiv::CUDADevice;
using primitiv::Graph;
using primitiv::Node;
using primitiv::Parameter;
using primitiv::SGDTrainer;
using primitiv::Shape;
using primitiv::initializers::Constant;
using primitiv::initializers::XavierUniform;
namespace F = primitiv::node_ops;
using namespace std;

namespace {

const unsigned NUM_TRAIN_SAMPLES = 60000;
const unsigned NUM_TEST_SAMPLES = 10000;
const unsigned NUM_INPUT_UNITS = 28 * 28;
const unsigned NUM_HIDDEN_UNITS = 300;
const unsigned NUM_OUTPUT_UNITS = 10;
const unsigned BATCH_SIZE = 200;
const unsigned NUM_TRAIN_BATCHES = NUM_TRAIN_SAMPLES / BATCH_SIZE;
const unsigned NUM_TEST_BATCHES = NUM_TEST_SAMPLES / BATCH_SIZE;
const unsigned MAX_EPOCH = 50;

// Helper function to load input images.
vector<float> load_images(const string &filename, const unsigned n) {
  ifstream ifs(filename, ios::binary);
  ifs.ignore(16);  // header
  const unsigned size = n * NUM_INPUT_UNITS;
  vector<unsigned char> buf(size);
  ifs.read(reinterpret_cast<char *>(&buf[0]), size);
  vector<float> ret(size);
  for (unsigned i = 0; i < size; ++i) ret[i] = buf[i] / 255.0;
  return ret;
}

// Helper function to load labels.
vector<char> load_labels(const string &filename, const unsigned n) {
  ifstream ifs(filename, ios::binary);
  ifs.ignore(8);  // header
  vector<char> ret(n);
  ifs.read(&ret[0], n);
  return ret;
}

}  // namespace

int main() {
  // Loads data
  vector<float> train_inputs = ::load_images(
      "mnist_data/train-images-idx3-ubyte", NUM_TRAIN_SAMPLES);
  vector<char> train_labels = ::load_labels(
      "mnist_data/train-labels-idx1-ubyte", NUM_TRAIN_SAMPLES);
  vector<float> test_inputs = ::load_images(
      "mnist_data/t10k-images-idx3-ubyte", NUM_TEST_SAMPLES);
  vector<char> test_labels = ::load_labels(
      "mnist_data/t10k-labels-idx1-ubyte", NUM_TEST_SAMPLES);

  // Uses GPU.
  CUDADevice dev(0);

  // Parameters
  Parameter pw1({NUM_HIDDEN_UNITS, NUM_INPUT_UNITS}, &dev, XavierUniform());
  Parameter pb1({NUM_HIDDEN_UNITS}, &dev, Constant(0));
  Parameter pw2({NUM_OUTPUT_UNITS, NUM_HIDDEN_UNITS}, &dev, XavierUniform());
  Parameter pb2({NUM_OUTPUT_UNITS}, &dev, Constant(0));

  // Trainer
  SGDTrainer trainer(.1);
  trainer.add_parameter(&pw1);
  trainer.add_parameter(&pb1);
  trainer.add_parameter(&pw2);
  trainer.add_parameter(&pb2);

  // Helper lambda to construct the predictor network.
  auto make_graph = [&](Graph &g, const vector<float> &inputs) {
    Node x = F::input(&g, &dev, Shape({NUM_INPUT_UNITS}, BATCH_SIZE), inputs);
    Node w1 = F::parameter(&g, &pw1);
    Node b1 = F::parameter(&g, &pb1);
    Node w2 = F::parameter(&g, &pw2);
    Node b2 = F::parameter(&g, &pb2);
    Node h = F::tanh(F::dot(w1, x) + b1);
    Node y = F::sigmoid(F::dot(w2, h) + b2);
    return y;
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
      vector<float> labels(BATCH_SIZE * NUM_OUTPUT_UNITS, 0);
      for (unsigned i = 0; i < BATCH_SIZE; ++i) {
        const unsigned id = ids[i + batch * BATCH_SIZE];
        copy(&train_inputs[id * NUM_INPUT_UNITS],
             &train_inputs[(id + 1) * NUM_INPUT_UNITS],
             &inputs[i * NUM_INPUT_UNITS]);
        labels[train_labels[id] + i * NUM_OUTPUT_UNITS] = 1;
      }

      // Constructs the graph.
      Graph g;
      Node y = make_graph(g, inputs);
      Node t = F::input(
          &g, &dev, Shape({NUM_OUTPUT_UNITS}, BATCH_SIZE), labels);
      Node diff = t - y;
      Node loss = diff * diff;
      Node sum_loss = F::batch_sum(loss) / BATCH_SIZE;

      // Forward, backward, and updates parameters.
      trainer.reset_gradients();
      g.forward(sum_loss);
      g.backward(sum_loss);
      trainer.update();
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
      Graph g;
      Node y = make_graph(g, inputs);

      // Gets outputs, argmax, and compares them with the label.
      vector<float> y_val = g.forward(y).to_vector();
      for (unsigned i = 0; i < BATCH_SIZE; ++i) {
        float maxval = -1e10;
        unsigned argmax = -1;
        for (unsigned j = 0; j < NUM_OUTPUT_UNITS; ++j) {
          float v = y_val[j + i * NUM_OUTPUT_UNITS];
          if (v > maxval) maxval = v, argmax = j;
        }
        if (argmax == test_labels[i + batch * BATCH_SIZE]) ++match;
      }
    }

    const float accuracy = 100.0 * match / NUM_TEST_SAMPLES;
    printf("epoch %d: accuracy: %.2f%%\n", epoch, accuracy);
  }

  return 0;
}
