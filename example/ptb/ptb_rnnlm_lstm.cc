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
#include <primitiv/primitiv_cuda.h>

using primitiv::initializers::Constant;
using primitiv::initializers::Uniform;
using primitiv::trainers::SGD;
namespace F = primitiv::operators;
using namespace primitiv;
using namespace std;

namespace {

static const unsigned NUM_HIDDEN_UNITS = 650;
static const unsigned BATCH_SIZE = 20;
static const unsigned MAX_EPOCH = 50;
static const float DROPOUT_RATE = 0.5;

// Gathers the set of words from space-separated corpus.
unordered_map<string, unsigned> make_vocab(const string &filename) {
  ifstream ifs(filename);
  if (!ifs.is_open()) {
    cerr << "File could not be opened: " << filename << endl;
    exit(1);
  }
  unordered_map<string, unsigned> vocab;
  string line, word;
  while (getline(ifs, line)) {
    line = "<bos>" + line + "<eos>";
    stringstream ss(line);
    while (getline(ss, word, ' ')) {
      if (vocab.find(word) == vocab.end()) {
        const unsigned id = vocab.size();
        vocab.emplace(make_pair(word, id));
      }
    }
  }
  return vocab;
}

// Generates word ID list using corpus and vocab.
vector<vector<unsigned>> load_corpus(
    const string &filename, const unordered_map<string, unsigned> &vocab) {
  ifstream ifs(filename);
  if (!ifs.is_open()) {
    cerr << "File could not be opened: " << filename << endl;
    exit(1);
  }
  vector<vector<unsigned>> corpus;
  string line, word;
  while (getline(ifs, line)) {
    line = "<bos>" + line + "<eos>";
    stringstream ss (line);
    vector<unsigned> sentence;
    while (getline(ss, word, ' ')) {
      sentence.emplace_back(vocab.at(word));
    }
    corpus.emplace_back(move(sentence));
  }
  return corpus;
}

// Counts output labels in the corpus.
unsigned count_labels(const vector<vector<unsigned>> &corpus) {
  unsigned ret = 0;
  for (const auto &sent :corpus) ret += sent.size() - 1;
  return ret;
}

// Extracts a minibatch from loaded corpus
vector<vector<unsigned>> make_batch(
    const vector<vector<unsigned>> &corpus,
    const vector<unsigned> &sent_ids,
    unsigned eos_id) {
  const unsigned batch_size = sent_ids.size();
  unsigned max_len = 0;
  for (const unsigned sid : sent_ids) {
    max_len = std::max<unsigned>(max_len, corpus[sid].size());
  }
  vector<vector<unsigned>> batch(max_len, vector<unsigned>(batch_size, eos_id));
  for (unsigned i = 0; i < batch_size; ++i) {
    const auto &sent = corpus[sent_ids[i]];
    for (unsigned j = 0; j < sent.size(); ++j) {
      batch[j][i] = sent[j];
    }
  }
  return batch;
}

// Affine transform:
//   y = W . x + b
template <typename Var>
class Affine {
  Parameter pw_, pb_;
  Var w_, b_;

public:
  Affine(const string &name,
      unsigned in_size, unsigned out_size, Trainer &trainer)
    : pw_(name + ".w", {out_size, in_size}, Uniform(-0.1, 0.1))
    , pb_(name + ".b", {out_size}, Constant(0)) {
      trainer.add_parameter(pw_);
      trainer.add_parameter(pb_);
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
class LSTM {
  unsigned out_size_;
  Parameter pwxh_, pwhh_, pbh_;
  Var wxh_, whh_, bh_, h_, c_;

public:
  LSTM(const string &name,
      unsigned in_size, unsigned out_size, Trainer &trainer)
    : out_size_(out_size)
    , pwxh_(name + ".wxh", {4 * out_size, in_size}, Uniform(-0.1, 0.1))
    , pwhh_(name + ".whh", {4 * out_size, out_size}, Uniform(-0.1, 0.1))
    , pbh_(name + ".bh", {4 * out_size}, Constant(0)) {
      trainer.add_parameter(pwxh_);
      trainer.add_parameter(pwhh_);
      trainer.add_parameter(pbh_);
    }

  // Initializes internal values.
  void init() {
    wxh_ = F::parameter<Var>(pwxh_);
    whh_ = F::parameter<Var>(pwhh_);
    bh_ = F::parameter<Var>(pbh_);
    h_ = c_ = F::zeros<Var>({out_size_});
  }

  // Forward one step.
  Var forward(const Var &x) {
    const Var u = F::matmul(wxh_, x) + F::matmul(whh_, h_) + bh_;
    const Var i = F::sigmoid(F::slice(u, 0, 0, out_size_));
    const Var f = F::sigmoid(F::slice(u, 0, out_size_, 2 * out_size_));
    const Var o = F::sigmoid(F::slice(u, 0, 2 * out_size_, 3 * out_size_));
    const Var j = F::tanh(F::slice(u, 0, 3 * out_size_, 4 * out_size_));
    c_ = i * j + f * c_;
    h_ = o * F::tanh(c_);
    return h_;
  }
};

// Language model using above LSTM.
template <typename Var>
class RNNLM {
  unsigned eos_id_;
  Parameter plookup_;
  LSTM<Var> rnn1_, rnn2_;
  Affine<Var> hy_;

public:
  RNNLM(unsigned vocab_size, unsigned eos_id, Trainer &trainer)
    : eos_id_(eos_id)
    , plookup_("lookup", {NUM_HIDDEN_UNITS, vocab_size}, Uniform(-0.1, 0.1))
    , rnn1_("rnn1", NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS, trainer)
    , rnn2_("rnn2", NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS, trainer)
    , hy_("hy", NUM_HIDDEN_UNITS, vocab_size, trainer) {
      trainer.add_parameter(plookup_);
    }

  // Forward function of RNNLM. Input data should be arranged below:
  // inputs = {
  //   {sent1_word1, sent2_word1, ..., sentN_word1},  // 1st input (<bos>)
  //   {sent1_word2, sent2_word2, ..., sentN_word2},  // 2nd input/1st output
  //   ...,
  //   {sent1_wordM, sent2_wordM, ..., sentN_wordM},  // last output (<eos>)
  // };
  vector<Var> forward(
      const vector<vector<unsigned>> &inputs, bool train) {
    const unsigned batch_size = inputs[0].size();
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
  const auto vocab = ::make_vocab("data/ptb.train.txt");
  cout << "#vocab: " << vocab.size() << endl;  // maybe 10001
  const unsigned eos_id = vocab.at("<eos>");

  // Loads all corpus.
  const auto train_corpus = ::load_corpus("data/ptb.train.txt", vocab);
  const auto valid_corpus = ::load_corpus("data/ptb.valid.txt", vocab);
  const unsigned num_train_sents = train_corpus.size();
  const unsigned num_valid_sents = valid_corpus.size();
  const unsigned num_train_labels = ::count_labels(train_corpus);
  const unsigned num_valid_labels = ::count_labels(valid_corpus);
  cout << "train: " << num_train_sents << " sentences, "
                    << num_train_labels << " labels" << endl;
  cout << "valid: " << num_valid_sents << " sentences, "
                    << num_valid_labels << " labels" << endl;

  // Uses GPU.
  devices::CUDA dev(0);
  Device::set_default(dev);
  Graph g;
  Graph::set_default(g);

  // Trainer.
  SGD trainer(1);
  //trainer.set_weight_decay(1e-6);
  trainer.set_gradient_clipping(5);

  // Our LM.
  ::RNNLM<Node> lm(vocab.size(), eos_id, trainer);

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
      const auto batch = ::make_batch(train_corpus, batch_ids, eos_id);

      g.clear();
      const auto outputs = lm.forward(batch, true);
      const auto loss = lm.loss(outputs, batch);
      train_loss += g.forward(loss).to_vector()[0] * batch_ids.size();

      trainer.reset_gradients();
      g.backward(loss);
      trainer.update();

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
      const auto batch = ::make_batch(valid_corpus, batch_ids, eos_id);

      g.clear();
      const auto outputs = lm.forward(batch, false);
      const auto loss = lm.loss(outputs, batch);
      valid_loss += g.forward(loss).to_vector()[0] * batch_ids.size();

      cout << ofs << '\r' << flush;
    }

    const float valid_ppl = std::exp(valid_loss / num_valid_labels);
    cout << "  valid ppl = " << valid_ppl << endl;

    if (valid_ppl < best_valid_ppl) {
      best_valid_ppl = valid_ppl;
      cout << "  BEST" << endl;
    } else {
      const float old_lr = trainer.get_learning_rate_scaling();
      const float new_lr = 0.5 * old_lr;
      trainer.set_learning_rate_scaling(new_lr);
      cout << "  learning rate scaled: " << old_lr << " -> " << new_lr << endl;
    }
  }

  return 0;
}
