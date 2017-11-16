// Sample code to train the SRU-based RNNLM using preprocessed Penn Treebank
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
using primitiv::optimizers::SGD;
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
  Affine(unsigned in_size, unsigned out_size, Optimizer &optimizer)
    : pw_({out_size, in_size}, Uniform(-0.1, 0.1))
    , pb_({out_size}, Constant(0)) {
      optimizer.add_parameter(pw_);
      optimizer.add_parameter(pb_);
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

// SRU cell.
// Formulation:
//   j[t] = W_xj . x[t]
//   f[t] = sigmoid(W_xf . x[t] + b_f)
//   r[t] = sigmoid(W_xr . x[t] + b_r)
//   c[t] = f[t] * c[t-1] + (1 - f[t]) * j[t]
//   h[t] = r[t] * tanh(c[t]) + (1 - r[t]) * x[t]
template <typename Var>
class SRU {
  unsigned out_size_;
  Parameter pw_, pbf_, pbr_;
  Var w_, bf_, br_;

public:
  SRU(unsigned in_size, unsigned out_size, Optimizer &optimizer)
    : out_size_(out_size)
    , pw_({3 * out_size, in_size}, Uniform(-0.1, 0.1))
    , pbf_({out_size}, Constant(0))
    , pbr_({out_size}, Constant(0)) {
      optimizer.add_parameter(pw_);
      optimizer.add_parameter(pbf_);
      optimizer.add_parameter(pbr_);
    }

  // Initializes internal values.
  void init() {
    w_ = F::parameter<Var>(pw_);
    bf_ = F::parameter<Var>(pbf_);
    br_ = F::parameter<Var>(pbr_);
  }

  // Forward.
  std::vector<Var> forward(const std::vector<Var> &xs) {
    const Var x = F::concat(xs, 1);
    const Var u = F::matmul(w_, x);
    const Var j = F::slice(u, 0, 0, out_size_);
    const Var f = F::sigmoid(
        F::slice(u, 0, out_size_, 2 * out_size_)
        + F::broadcast(bf_, 1, xs.size()));
    const Var r = F::sigmoid(
        F::slice(u, 0, 2 * out_size_, 3 * out_size_)
        + F::broadcast(bf_, 1, xs.size()));
    Var c = F::zeros<Var>({out_size_});
    std::vector<Var> hs;
    for (unsigned i = 0; i < xs.size(); ++i) {
      const Var ji = F::slice(j, 1, i, i + 1);
      const Var fi = F::slice(f, 1, i, i + 1);
      const Var ri = F::slice(r, 1, i, i + 1);
      c = fi * c + (1 - fi) * ji;
      hs.emplace_back(ri * F::tanh(c) + (1 - ri) * xs[i]);
    }
    return hs;
  }
};

// Language model using above SRU.
template <typename Var>
class RNNLM {
  unsigned eos_id_;
  Parameter plookup_;
  SRU<Var> rnn1_, rnn2_;
  Affine<Var> hy_;

public:
  RNNLM(unsigned vocab_size, unsigned eos_id, Optimizer &optimizer)
    : eos_id_(eos_id)
    , plookup_({NUM_HIDDEN_UNITS, vocab_size}, Uniform(-0.1, 0.1))
    , rnn1_(NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS, optimizer)
    , rnn2_(NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS, optimizer)
    , hy_(NUM_HIDDEN_UNITS, vocab_size, optimizer) {
      optimizer.add_parameter(plookup_);
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

    vector<Var> xs;
    for (unsigned i = 0; i < inputs.size() - 1; ++i) {
      xs.emplace_back(
          F::dropout(F::pick(lookup, inputs[i], 1), DROPOUT_RATE, train));
    }
    vector<Var> hs1 = rnn1_.forward(xs);
    for (unsigned i = 0; i < inputs.size() - 1; ++i) {
      hs1[i] = F::dropout(hs1[i], DROPOUT_RATE, train);
    }
    vector<Var> hs2 = rnn2_.forward(hs1);
    vector<Var> outputs;
    for (unsigned i = 0; i < inputs.size() - 1; ++i) {
      outputs.emplace_back(
          hy_.forward(F::dropout(hs2[i], DROPOUT_RATE, train)));
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

  // Optimizer.
  SGD optimizer(1);
  //optimizer.set_weight_decay(1e-6);
  optimizer.set_gradient_clipping(5);

  // Our LM.
  ::RNNLM<Node> lm(vocab.size(), eos_id, optimizer);

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
      const auto batch = ::make_batch(valid_corpus, batch_ids, eos_id);

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
