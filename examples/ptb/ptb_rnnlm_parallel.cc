// Sample code to train the RNNLM with data-parallel computation using
// preprocessed Penn Treebank dataset:
//   http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
//
// Usage:
//   Run 'download_data.sh' in the same directory before using this code.
// g++
//   -std=c++11
//   -I/path/to/primitiv/includes (typically -I../..)
//   -L/path/to/primitiv/libs     (typically -L../../build/primitiv)
//   ptb_rnnlm_parallel.cc -lprimitiv

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <primitiv/primitiv.h>

#include "utils.h"

using primitiv::initializers::XavierUniform;
using primitiv::optimizers::Adam;
namespace F = primitiv::functions;
using namespace primitiv;
using namespace std;

namespace {

static const unsigned NUM_HIDDEN_UNITS = 256;
static const unsigned BATCH_SIZE = 128;
static const unsigned MAX_EPOCH = 100;

class RNNLM : public Model {
public:
  RNNLM(unsigned vocab_size, Device *device)
    : pwlookup_({NUM_HIDDEN_UNITS, vocab_size}, XavierUniform(), device)
    , pwxs_({NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS}, XavierUniform(), device)
    , pwsy_({vocab_size, NUM_HIDDEN_UNITS}, XavierUniform(), device)
    , device_(device) {
      add("pwlookup", pwlookup_);
      add("pwxs", pwxs_);
      add("pwsy", pwsy_);
    }

  // Forward function of RNNLM. Input data should be arranged below:
  // inputs = {
  //   {sent1_word1, sent2_word1, ..., sentN_word1},  // 1st input (<s>)
  //   {sent1_word2, sent2_word2, ..., sentN_word2},  // 2nd input/1st output
  //   ...,
  //   {sent1_wordM, sent2_wordM, ..., sentN_wordM},  // last output (<s>)
  // };
  vector<Node> forward(const vector<vector<unsigned>> &inputs) {
    const unsigned batch_size = inputs[0].size();
    Node wlookup = F::parameter<Node>(pwlookup_);
    Node wxs = F::parameter<Node>(pwxs_);
    Node wsy = F::parameter<Node>(pwsy_);
    Node s = F::zeros<Node>(Shape({NUM_HIDDEN_UNITS}, batch_size), device_);
    vector<Node> outputs;
    for (unsigned i = 0; i < inputs.size() - 1; ++i) {
      Node w = F::pick(wlookup, inputs[i], 1);
      Node x = w + s;
      Node s = F::sigmoid(F::matmul(wxs, x));
      outputs.emplace_back(F::matmul(wsy, s));
    }
    return outputs;
  }

  // Loss function.
  Node forward_loss(
      const vector<Node> &outputs, const vector<vector<unsigned>> &inputs) {
    vector<Node> losses;
    for (unsigned i = 0; i < outputs.size(); ++i) {
      losses.emplace_back(
          F::softmax_cross_entropy(outputs[i], inputs[i + 1], 0));
    }
    return F::batch::mean(F::sum(losses));
  }

private:
  Parameter pwlookup_;
  Parameter pwxs_;
  Parameter pwsy_;
  Device *device_;
};

}  // namespace

int main() {
  // Loads vocab.
  const auto vocab = utils::make_vocab("data/ptb.train.txt");
  cout << "#vocab: " << vocab.size() << endl;  // maybe 10000
  const unsigned eos_id = vocab.at("<s>");

  // Loads all corpus.
  const auto train_corpus = utils::load_corpus("data/ptb.train.txt", vocab);
  const auto valid_corpus = utils::load_corpus("data/ptb.valid.txt", vocab);
  const unsigned num_train_sents = train_corpus.size();
  const unsigned num_valid_sents = valid_corpus.size();
  const unsigned num_train_labels = utils::count_labels(train_corpus);
  const unsigned num_valid_labels = utils::count_labels(valid_corpus);
  cout << "train: " << num_train_sents << " sentences, "
                    << num_train_labels << " labels" << endl;
  cout << "valid: " << num_valid_sents << " sentences, "
                    << num_valid_labels << " labels" << endl;

  devices::Naive dev1;  //devices::CUDA dev1(0);
  devices::Naive dev2;  //devices::CUDA dev2(1);
  Graph g;
  Graph::set_default(g);

  // Our LMs.
  ::RNNLM lm1(vocab.size(), &dev1);
  ::RNNLM lm2(vocab.size(), &dev2);
  lm1.copy(lm2);

  // Optimizer.
  Adam optimizer;
  optimizer.set_weight_decay(1e-6);
  optimizer.set_gradient_clipping(5);
  optimizer.add(lm1);

  // Batch randomizer.
  random_device rd;
  mt19937 rng(rd());

  // Sentence IDs.
  vector<unsigned> train_ids(num_train_sents);
  vector<unsigned> valid_ids(num_valid_sents);
  iota(begin(train_ids), end(train_ids), 0);
  iota(begin(valid_ids), end(valid_ids), 0);

  // Train/valid loop.
  for (unsigned epoch = 0; epoch < MAX_EPOCH; ++epoch) {
    cout << "epoch " << (epoch + 1) << '/' << MAX_EPOCH << ':' << endl;
    // Shuffles train sentence IDs.
    shuffle(begin(train_ids), end(train_ids), rng);

    // Training.
    float train_loss = 0;
    for (unsigned ofs = 0; ofs < num_train_sents; ofs += BATCH_SIZE) {
      const vector<unsigned> batch_ids1(
          begin(train_ids) + ofs,
          begin(train_ids) + std::min<unsigned>(
            ofs + BATCH_SIZE / 2, num_train_sents));
      const auto batch1 = utils::make_batch(train_corpus, batch_ids1, eos_id);

      g.clear();
      const auto outputs1 = lm1.forward(batch1);
      const auto loss1 = lm1.forward_loss(outputs1, batch1);
      train_loss += loss1.to_float() * batch_ids1.size();
      lm1.reset_gradients();
      loss1.backward();

      if (ofs + BATCH_SIZE / 2 < num_train_sents) {
        const vector<unsigned> batch_ids2(
            begin(train_ids) + ofs + BATCH_SIZE / 2,
            begin(train_ids) + std::min<unsigned>(
              ofs + BATCH_SIZE, num_train_sents));
        const auto batch2 = utils::make_batch(train_corpus, batch_ids2, eos_id);

        const auto outputs2 = lm2.forward(batch2);
        const auto loss2 = lm2.forward_loss(outputs2, batch2);
        train_loss += loss2.to_float() * batch_ids2.size();

        lm2.reset_gradients();
        loss2.backward();
        lm1.add_gradients(lm2);
      }

      optimizer.update();
      lm1.copy(lm2);

      cout << ofs << '\r' << flush;
    }

    const float train_ppl = std::exp(train_loss / num_train_labels);
    cout << "  train ppl = " << train_ppl << endl;

    // Validation.
    float valid_loss = 0;
    for (unsigned ofs = 0; ofs < num_valid_sents; ofs += BATCH_SIZE) {
      const vector<unsigned> batch_ids1(
          begin(valid_ids) + ofs,
          begin(valid_ids) + std::min<unsigned>(
            ofs + BATCH_SIZE / 2, num_valid_sents));
      const auto batch1 = utils::make_batch(valid_corpus, batch_ids1, eos_id);

      g.clear();
      const auto outputs1 = lm1.forward(batch1);
      const auto loss1 = lm1.forward_loss(outputs1, batch1);
      valid_loss += loss1.to_float() * batch_ids1.size();

      if (ofs + BATCH_SIZE / 2 < num_valid_sents) {
        const vector<unsigned> batch_ids2(
            begin(valid_ids) + ofs + BATCH_SIZE / 2,
            begin(valid_ids) + std::min<unsigned>(
              ofs + BATCH_SIZE, num_valid_sents));
        const auto batch2 = utils::make_batch(valid_corpus, batch_ids2, eos_id);

        const auto outputs2 = lm2.forward(batch2);
        const auto loss2 = lm2.forward_loss(outputs2, batch2);
        valid_loss += loss2.to_float() * batch_ids2.size();
      }

      cout << ofs << '\r' << flush;
    }

    const float valid_ppl = std::exp(valid_loss / num_valid_labels);
    cout << "  valid ppl = " << valid_ppl << endl;
  }

  return 0;
}
