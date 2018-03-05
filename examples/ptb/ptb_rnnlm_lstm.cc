// Sample code to train the LSTM-based RNNLM using preprocessed Penn Treebank
// dataset:
//   http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
//
// Usage:
//   Run 'download_data.sh' in the same directory before using this code.
// g++
//   -std=c++11
//   -I/path/to/primitiv/includes (typically -I../..)
//   -L/path/to/primitiv/libs     (typically -L../../build/primitiv)
//   ptb_rnnlm_lstm.cc -lprimitiv

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

using primitiv::initializers::Constant;
using primitiv::initializers::Uniform;
using primitiv::optimizers::SGD;
namespace F = primitiv::functions;
using namespace primitiv;
using namespace std;

namespace {

static const unsigned NUM_HIDDEN_UNITS = 650;
static const unsigned BATCH_SIZE = 20;
static const unsigned MAX_EPOCH = 50;
static const float DROPOUT_RATE = 0.5;

// Affine transform:
//   y = W . x + b
template <typename Var>
class Affine : public Model {
  Parameter pw_, pb_;
  Var w_, b_;

public:
  Affine(unsigned in_size, unsigned out_size)
    : pw_({out_size, in_size}, Uniform(-0.1, 0.1))
    , pb_({out_size}, Constant(0)) {
      add("pw", pw_);
      add("pb", pb_);
    }

  // Initializes internal values.
  void init() {
    w_ = F::parameter<Var>(pw_);
    b_ = F::parameter<Var>(pb_);
  }

  // Applies transform.
  Var forward(const Var &x) {
    return F::matmul(w_, x) + b_;
  }
};

// LSTM with input/forget/output gates and no peepholes.
// Formulation:
//   i = sigmoid(W_xi . x[t] + W_hi . h[t-1] + b_i)
//   f = sigmoid(W_xf . x[t] + W_hf . h[t-1] + b_f)
//   o = sigmoid(W_xo . x[t] + W_ho . h[t-1] + b_o)
//   j = tanh   (W_xj . x[t] + W_hj . h[t-1] + b_j)
//   c[t] = i * j + f * c[t-1]
//   h[t] = o * tanh(c[t])
template <typename Var>
class LSTM : public Model {
  unsigned out_size_;
  Parameter pw_, pb_;
  Var w_, b_, h_, c_;

public:
  LSTM(unsigned in_size, unsigned out_size)
    : out_size_(out_size)
    , pw_({4 * out_size, in_size + out_size}, Uniform(-0.1, 0.1))
    , pb_({4 * out_size}, Constant(0)) {
      add("pw", pw_);
      add("pb", pb_);
    }

  // Initializes internal values.
  void init() {
    w_ = F::parameter<Var>(pw_);
    b_ = F::parameter<Var>(pb_);
    h_ = c_ = F::zeros<Var>({out_size_});
  }

  // Forward one step.
  Var forward(const Var &x) {
    const Var u = F::matmul(w_, F::concat({x, h_}, 0)) + b_;
    const std::vector<Var> v = F::split(u, 0, 4);
    const Var i = F::sigmoid(v[0]);
    const Var f = F::sigmoid(v[1]);
    const Var o = F::sigmoid(v[2]);
    const Var j = F::tanh(v[3]);
    c_ = i * j + f * c_;
    h_ = o * F::tanh(c_);
    return h_;
  }
};

// Language model using above LSTM.
template <typename Var>
class RNNLM : public Model {
  Parameter plookup_;
  LSTM<Var> rnn1_, rnn2_;
  Affine<Var> hy_;

public:
  RNNLM(unsigned vocab_size)
    : plookup_({NUM_HIDDEN_UNITS, vocab_size}, Uniform(-0.1, 0.1))
    , rnn1_(NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS)
    , rnn2_(NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS)
    , hy_(NUM_HIDDEN_UNITS, vocab_size) {
      add("plookup", plookup_);
      add("rnn1", rnn1_);
      add("rnn2", rnn2_);
      add("hy", hy_);
    }

  // Forward function of RNNLM. Input data should be arranged below:
  // inputs = {
  //   {sent1_word1, sent2_word1, ..., sentN_word1},  // 1st input (<bos>)
  //   {sent1_word2, sent2_word2, ..., sentN_word2},  // 2nd input/1st output
  //   ...,
  //   {sent1_wordM, sent2_wordM, ..., sentN_wordM},  // last output (<eos>)
  // };
  vector<Var> forward(const vector<vector<unsigned>> &inputs, bool train) {
    Var lookup = F::parameter<Var>(plookup_);
    rnn1_.init();
    rnn2_.init();
    hy_.init();

    vector<Var> outputs;
    for (unsigned i = 0; i < inputs.size() - 1; ++i) {
      Var x = F::pick(lookup, inputs[i], 1);
      x = F::dropout(x, DROPOUT_RATE, train);
      Var h1 = rnn1_.forward(x);
      h1 = F::dropout(h1, DROPOUT_RATE, train);
      Var h2 = rnn2_.forward(h1);
      h2 = F::dropout(h2, DROPOUT_RATE, train);
      outputs.emplace_back(hy_.forward(h2));
    }
    return outputs;
  }

  // Loss function.
  Var loss(
      const vector<Var> &outputs, const vector<vector<unsigned>> &inputs) {
    vector<Var> losses;
    for (unsigned i = 0; i < outputs.size(); ++i) {
      losses.emplace_back(
          F::softmax_cross_entropy(outputs[i], inputs[i + 1], 0));
    }
    return F::batch::mean(F::sum(losses));
  }
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

  devices::CUDA dev(0);  // devices::CUDA dev(0);
  Device::set_default(dev);
  Graph g;
  Graph::set_default(g);

  // Our LM.
  ::RNNLM<Node> lm(vocab.size());

  // Optimizer.
  SGD optimizer(1);
  //optimizer.set_weight_decay(1e-6);
  optimizer.set_gradient_clipping(5);
  optimizer.add(lm);

  // Batch randomizer.
  random_device rd;
  mt19937 rng(rd());

  // Sentence IDs.
  vector<unsigned> train_ids(num_train_sents);
  vector<unsigned> valid_ids(num_valid_sents);
  iota(begin(train_ids), end(train_ids), 0);
  iota(begin(valid_ids), end(valid_ids), 0);

  float best_valid_ppl = 1e10;

  // Train/valid loop.
  for (unsigned epoch = 0; epoch < MAX_EPOCH; ++epoch) {
    cout << "epoch " << (epoch + 1) << '/' << MAX_EPOCH << ':' << endl;
    // Shuffles train sentence IDs.
    shuffle(begin(train_ids), end(train_ids), rng);

    // Training.
    float train_loss = 0;
    for (unsigned ofs = 0; ofs < num_train_sents; ofs += BATCH_SIZE) {
      const vector<unsigned> batch_ids(
          begin(train_ids) + ofs,
          begin(train_ids) + std::min<unsigned>(
            ofs + BATCH_SIZE, num_train_sents));
      const auto batch = utils::make_batch(train_corpus, batch_ids, eos_id);

      g.clear();
      const auto outputs = lm.forward(batch, true);
      const auto loss = lm.loss(outputs, batch);
      train_loss += loss.to_float() * batch_ids.size();

      optimizer.reset_gradients();
      loss.backward();
      optimizer.update();

      cout << ofs << '\r' << flush;
    }

    const float train_ppl = std::exp(train_loss / num_train_labels);
    cout << "  train ppl = " << train_ppl << endl;

    // Validation.
    float valid_loss = 0;
    for (unsigned ofs = 0; ofs < num_valid_sents; ofs += BATCH_SIZE) {
      const vector<unsigned> batch_ids(
          begin(valid_ids) + ofs,
          begin(valid_ids) + std::min<unsigned>(
            ofs + BATCH_SIZE, num_valid_sents));
      const auto batch = utils::make_batch(valid_corpus, batch_ids, eos_id);

      g.clear();
      const auto outputs = lm.forward(batch, false);
      const auto loss = lm.loss(outputs, batch);
      valid_loss += loss.to_float() * batch_ids.size();

      cout << ofs << '\r' << flush;
    }

    const float valid_ppl = std::exp(valid_loss / num_valid_labels);
    cout << "  valid ppl = " << valid_ppl << endl;

    if (valid_ppl < best_valid_ppl) {
      best_valid_ppl = valid_ppl;
      cout << "  BEST" << endl;
    } else {
      const float old_lr = optimizer.get_learning_rate_scaling();
      const float new_lr = 0.5 * old_lr;
      optimizer.set_learning_rate_scaling(new_lr);
      cout << "  learning rate scaled: " << old_lr << " -> " << new_lr << endl;
    }
  }

  return 0;
}
