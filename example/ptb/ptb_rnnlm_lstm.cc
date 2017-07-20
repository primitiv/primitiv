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
using primitiv::initializers::XavierUniform;
using primitiv::trainers::Adam;
namespace F = primitiv::node_ops;
using namespace primitiv;
using namespace std;

namespace {

static const unsigned NUM_EMBED_UNITS = 512;
static const unsigned NUM_HIDDEN_UNITS = 512;
static const unsigned BATCH_SIZE = 64;
static const unsigned MAX_EPOCH = 100;
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
    line = "<s>" + line + "<s>";
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
    line = "<s>" + line + "<s>";
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

// Hand-written LSTM with input/forget/output gates and no peepholes.
// Formulation:
//   i = sigmoid(W_xi . x[t] + W_hi . h[t-1] + b_i)
//   f = sigmoid(W_xf . x[t] + W_hf . h[t-1] + b_f)
//   o = sigmoid(W_xo . x[t] + W_ho . h[t-1] + b_o)
//   j = tanh   (W_xj . x[t] + W_hj . h[t-1] + b_j)
//   c[t] = i * j + f * c[t-1]
//   h[t] = o * tanh(c[t])
class LSTM {
  public:
    LSTM(unsigned in_size, unsigned out_size, Trainer &trainer)
      : out_size_(out_size)
      , pwxh_("wxh", {4 * out_size, in_size}, XavierUniform())
      , pwhh_("whh", {4 * out_size, out_size}, XavierUniform())
      , pbh_("bh", {4 * out_size}, Constant(0)) {
        trainer.add_parameter(pwxh_);
        trainer.add_parameter(pwhh_);
        trainer.add_parameter(pbh_);
      }

    // Initializes internal values.
    void init() {
      wxh_ = F::input(pwxh_);
      whh_ = F::input(pwhh_);
      bh_ = F::input(pbh_);
      h_ = c_ = F::zeros({out_size_});
    }

    // Forward one step.
    Node forward(const Node &x) {
      const Node u = F::matmul(wxh_, x) + F::matmul(whh_, h_) + bh_;
      const Node i = F::sigmoid(F::slice(u, 0, 0, out_size_));
      const Node f = F::sigmoid(F::slice(u, 0, out_size_, 2 * out_size_));
      const Node o = F::sigmoid(F::slice(u, 0, 2 * out_size_, 3 * out_size_));
      const Node j = F::tanh(F::slice(u, 0, 3 * out_size_, 4 * out_size_));
      c_ = i * j + f * c_;
      h_ = o * F::tanh(c_);
      return h_;
    }

  private:
    unsigned out_size_;
    Parameter pwxh_, pwhh_, pbh_;
    Node wxh_, whh_, bh_, h_, c_;
};

// Language model using above LSTM.
class RNNLM {
public:
  RNNLM(unsigned vocab_size, unsigned eos_id, Trainer &trainer)
    : eos_id_(eos_id)
    , plookup_("lookup", {NUM_EMBED_UNITS, vocab_size}, XavierUniform())
    , pwhy_("why", {vocab_size, NUM_HIDDEN_UNITS}, XavierUniform())
    , pby_("by", {vocab_size}, Constant(0))
    , lstm_(NUM_EMBED_UNITS, NUM_HIDDEN_UNITS, trainer) {
      trainer.add_parameter(plookup_);
      trainer.add_parameter(pwhy_);
      trainer.add_parameter(pby_);
    }

  // Forward function of RNNLM. Input data should be arranged below:
  // inputs = {
  //   {sent1_word1, sent2_word1, ..., sentN_word1},  // 1st input (<s>)
  //   {sent1_word2, sent2_word2, ..., sentN_word2},  // 2nd input/1st output
  //   ...,
  //   {sent1_wordM, sent2_wordM, ..., sentN_wordM},  // last output (<s>)
  // };
  vector<Node> forward(
      const vector<vector<unsigned>> &inputs, bool train) {
    const unsigned batch_size = inputs[0].size();
    Node lookup = F::input(plookup_);
    Node why = F::input(pwhy_);
    Node by = F::input(pby_);
    lstm_.init();

    vector<Node> outputs;
    for (unsigned i = 0; i < inputs.size() - 1; ++i) {
      Node x = F::pick(lookup, 1, inputs[i]);
      x = F::dropout(x, DROPOUT_RATE, train);
      Node h = lstm_.forward(x);
      h = F::dropout(h, DROPOUT_RATE, train);
      outputs.emplace_back(F::matmul(why, h) + by);
    }
    return outputs;
  }

  // Loss function.
  Node forward_loss(
      const vector<Node> &outputs, const vector<vector<unsigned>> &inputs) {
    vector<Node> losses;
    for (unsigned i = 0; i < outputs.size(); ++i) {
      losses.emplace_back(
          F::softmax_cross_entropy(outputs[i], 0, inputs[i + 1]));
    }
    return F::batch::mean(F::sum(losses));
  }

private:
  unsigned eos_id_;
  Parameter plookup_, pwhy_, pby_;
  LSTM lstm_;
};

}  // namespace

int main() {
  // Loads vocab.
  const auto vocab = ::make_vocab("data/ptb.train.txt");
  cout << "#vocab: " << vocab.size() << endl;  // maybe 10000
  const unsigned eos_id = vocab.at("<s>");

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
  CUDADevice dev(0);
  Device::set_default_device(dev);

  // Trainer.
  Adam trainer;
  trainer.set_weight_decay(1e-6);
  trainer.set_gradient_clipping(5);

  // Our LM.
  ::RNNLM lm(vocab.size(), eos_id, trainer);

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
      const vector<unsigned> batch_ids(
          begin(train_ids) + ofs,
          begin(train_ids) + std::min<unsigned>(
            ofs + BATCH_SIZE, num_train_sents));
      const auto batch = ::make_batch(train_corpus, batch_ids, eos_id);
      trainer.reset_gradients();
      Graph g;
      Graph::set_default_graph(g);
      const auto outputs = lm.forward(batch, true);
      const auto loss = lm.forward_loss(outputs, batch);
      train_loss += g.forward(loss).to_vector()[0] * batch_ids.size();
      g.backward(loss);
      trainer.update(1);
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
      Graph g;
      Graph::set_default_graph(g);
      const auto outputs = lm.forward(batch, false);
      const auto loss = lm.forward_loss(outputs, batch);
      valid_loss += g.forward(loss).to_vector()[0] * batch_ids.size();
      cout << ofs << '\r' << flush;
    }
    const float valid_ppl = std::exp(valid_loss / num_valid_labels);
    cout << "  valid ppl = " << valid_ppl << endl;
  }

  return 0;
}
