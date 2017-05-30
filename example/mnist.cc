#include <algorithm>
#include <random>
#include <fstream>
#include <iostream>
#include <vector>

#define PRIMITIV_USE_CUDA
#include <primitiv/primitiv.h>

using namespace std;
namespace I = primitiv::initializers;
namespace F = primitiv::node_ops;
using Trainer = primitiv::SGDTrainer;
using primitiv::Graph;
using primitiv::Node;
using primitiv::Parameter;
using primitiv::Shape;

int main() {
  const unsigned TRAIN_N = 60000;
  const unsigned TEST_N = 10000;

  vector<float> train_inputs(TRAIN_N*28*28);
  vector<float> train_labels(TRAIN_N*10, 0);
  vector<float> test_inputs(TEST_N*28*28);
  vector<float> test_labels(TEST_N*10, 0);

  // load data
  ifstream ifs_train_inputs("train-images-idx3-ubyte", ios::binary);
  ifstream ifs_train_labels("train-labels-idx1-ubyte", ios::binary);
  ifstream ifs_test_inputs("t10k-images-idx3-ubyte", ios::binary);
  ifstream ifs_test_labels("t10k-labels-idx1-ubyte", ios::binary);
  ifs_train_inputs.ignore(16);
  ifs_train_labels.ignore(8);
  ifs_test_inputs.ignore(16);
  ifs_test_labels.ignore(8);
  vector<unsigned char> buf(TRAIN_N*28*28);
  ifs_train_inputs.read(reinterpret_cast<char *>(&buf[0]), TRAIN_N*28*28);
  for (unsigned i = 0; i < TRAIN_N*28*28; ++i) train_inputs[i] = buf[i] / 128. - 1;
  ifs_train_labels.read(reinterpret_cast<char *>(&buf[0]), TRAIN_N);
  for (unsigned i = 0; i < TRAIN_N; ++i) train_labels[i*10+buf[i]] = 1.;
  ifs_test_inputs.read(reinterpret_cast<char *>(&buf[0]), TEST_N*28*28);
  for (unsigned i = 0; i < TEST_N*28*28; ++i) test_inputs[i] = buf[i] / 128. - 1;
  ifs_test_labels.read(reinterpret_cast<char *>(&buf[0]), TEST_N);
  for (unsigned i = 0; i < TEST_N; ++i) test_labels[i*10+buf[i]] = 1.;

  primitiv::CUDADevice dev(6);

  Parameter pw1({300, 28*28}, &dev, I::XavierUniform());
  Parameter pb1({300}, &dev, I::Constant(0));
  Parameter pw2({10, 300}, &dev, I::XavierUniform());
  Parameter pb2({10}, &dev, I::Constant(0));

  Trainer trainer(.1);
  trainer.add_parameter(&pw1);
  trainer.add_parameter(&pb1);
  trainer.add_parameter(&pw2);
  trainer.add_parameter(&pb2);

  const unsigned BS = 200;
  const unsigned BN = TRAIN_N / BS;
  
  auto make_network = [&](Graph &g, const vector<float> inputs) {
    Node x = F::input(g, dev, Shape({28*28}, BS), inputs);
    Node w1 = F::parameter(g, pw1);
    Node b1 = F::parameter(g, pb1);
    Node w2 = F::parameter(g, pw2);
    Node b2 = F::parameter(g, pb2);
    Node h = F::tanh(F::dot(w1, x) + b1);
    Node y = F::sigmoid(F::dot(w2, h) + b2);
    return y;
  };

  mt19937 rng;
  vector<unsigned> ids(TRAIN_N);
  iota(begin(ids), end(ids), 0);

  for (unsigned epoch = 0; epoch < 20; ++epoch) {
    shuffle(begin(ids), end(ids), rng);

    for (unsigned batch = 0; batch < BN; ++batch) {
      vector<float> inputs(BS*28*28);
      vector<float> labels(BS*10);
      for (unsigned i = 0; i < BS; ++i) {
        const unsigned id = ids[i + BS*batch];
        copy(&train_inputs[id*28*28], &train_inputs[(id+1)*28*28], &inputs[i*28*28]);
        copy(&train_labels[id*10], &train_labels[(id+1)*10], &labels[i*10]);
      }

      Graph g;
      Node y = make_network(g, inputs);
      
      Node t = F::input(g, dev, Shape({10}, BS), labels);
      Node diff = t - y;
      Node loss = diff * diff;
      Node sum_loss = F::batch_sum(loss) / BS * pow(0.8, epoch);

      const float loss_val = g.forward(sum_loss).to_vector()[0];
      //cout << "epoch " << epoch << ": batch " << batch << ": loss: " << loss_val << endl;

      trainer.reset_gradients();
      g.backward(sum_loss);
      trainer.update();
    }

    const unsigned BN_TEST = TEST_N / BS;
    unsigned match = 0;

    for (unsigned batch = 0; batch < BN_TEST; ++batch) {
      vector<float> inputs(BS*28*28);
      vector<float> labels(BS*10);
      for (unsigned i = 0; i < BS; ++i) {
        const unsigned id = i + BS*batch;
        copy(&test_inputs[id*28*28], &test_inputs[(id+1)*28*28], &inputs[i*28*28]);
        copy(&test_labels[id*10], &test_labels[(id+1)*10], &labels[i*10]);
      }

      Graph g;
      Node y = make_network(g, inputs);
      vector<float> y_val = g.forward(y).to_vector();
      for (unsigned i = 0; i < BS; ++i) {
        float maxval = -10000;
        float argmax = -1;
        for (unsigned j = 0; j < 10; ++j) {
          float v = y_val[j + i*10];
          if (v > maxval) maxval = v, argmax = j;
        }
        float label = 0;
        for (; labels[label + i*10] == 0; ++label);
        //cout << "label: " << label << ", argmax: " << argmax << endl;
        if (label == argmax) ++match;
      }
    }

    printf("epoch %d: accuracy: %.2f%%\n", epoch, 100.*(float)match/TEST_N);
  }

  return 0;
}
