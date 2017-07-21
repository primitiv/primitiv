// Sample code to train the encoder-decoder model using small English-Japanese
// parallel corpora.
//
// Model detail:
//   Sutskever et al., 2014.
//   Sequence to Sequence Learning with Neural Networks.
//   https://arxiv.org/abs/1409.3215
//
// Corpora detail:
//   https://github.com/odashi/small_parallel_enja
//
// Usage:
//   Run 'download_data.sh' in the same directory before using this code.
// g++
//   -std=c++11
//   -I/path/to/primitiv/includes (typically -I../..)
//   -L/path/to/primitiv/libs     (typically -L../../build/primitiv)
//   encdec.cc -lprimitiv

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <primitiv/primitiv.h>
#include <primitiv/primitiv_cuda.h>

using primitiv::trainers::Adam;
namespace F = primitiv::node_ops;
namespace I = primitiv::initializers;
using namespace primitiv;
using namespace std;

namespace {

static const unsigned NUM_EMBED_UNITS = 512;
static const unsigned NUM_HIDDEN_UNITS = 512;
static const unsigned BATCH_SIZE = 64;
static const unsigned MAX_EPOCH = 100;
static const float DROPOUT_RATE = 0.5;

// Helper to open fstream
template <class FStreamT>
void open_file(const string &path, FStreamT &fs) {
  fs.open(path);
  if (!fs.is_open()) {
    cerr << "File could not be opened: " + path << endl;
    exit(1);
  }
}

// Gathers the set of words from space-separated corpus and makes a vocabulary.
unordered_map<string, unsigned> make_vocab(const string &path, unsigned size) {
  if (size < 3) {
    cerr << "vocab size should be equal-to or greater-than 3." << endl;
    exit(1);
  }
  ifstream ifs;
  ::open_file(path, ifs);

  // Counts all word existences.
  unordered_map<string, unsigned> freq;
  string line, word;
  while (getline(ifs, line)) {
    stringstream ss(line);
    while (getline(ss, word, ' ')) ++freq[word];
  }

  // Sorting.
  using freq_t = pair<string, unsigned>;
  auto cmp = [](const freq_t &a, const freq_t &b) {
    return a.second < b.second;
  };
  priority_queue<freq_t, vector<freq_t>, decltype(cmp)> q(cmp);
  for (const auto &x : freq) q.push(x);

  // Chooses top size-3 frequent words to make the vocabulary.
  unordered_map<string, unsigned> vocab;
  vocab.insert(make_pair("<unk>", 0));
  vocab.insert(make_pair("<bos>", 1));
  vocab.insert(make_pair("<eos>", 2));
  for (unsigned i = 3; i < size; ++i) {
    vocab.insert(make_pair(q.top().first, i));
    q.pop();
  }

  return vocab;
}

// Generates word ID list using a corpus and a vocab.
// All out-of-vocab words are replaced to <unk>.
vector<vector<unsigned>> load_corpus(
    const string &path, const unordered_map<string, unsigned> &vocab) {
  const unsigned unk_id = vocab.at("<unk>");
  ifstream ifs;
  ::open_file(path, ifs);
  vector<vector<unsigned>> corpus;
  string line, word;
  while (getline(ifs, line)) {
    line = "<bos> " + line + " <eos>";
    stringstream ss (line);
    vector<unsigned> sentence;
    while (getline(ss, word, ' ')) {
      const auto it = vocab.find(word);
      if (it != vocab.end()) sentence.emplace_back(it->second);
      else sentence.emplace_back(unk_id);
    }
    corpus.emplace_back(move(sentence));
  }
  return corpus;
}

// Counts output labels in the corpus.
unsigned count_labels(const vector<vector<unsigned>> &corpus) {
  unsigned ret = 0;
  for (const auto &sent :corpus) ret += sent.size() - 1;  // w/o <bos>
  return ret;
}

// Extracts a minibatch from loaded corpus
// NOTE(odashi):
// Lengths of all sentences are adjusted to the maximum one in the minibatch.
// All additional subsequences are filled by <eos>. E.g.,
//   input: {
//     {<bos>, w1, <eos>},
//     {<bos>, w1, w2, w3, w4, <eos>},
//     {<bos>, w1, w2, <eos>},
//     {<bos>, w1, w2, w3, <eos>},
//   }
//   output: {
//     {<bos>, <bos>, <bos>, <bos>},
//     {   w1,    w1,    w1,    w1},
//     {<eos>,    w2,    w2,    w2},
//     {<eos>,    w3, <eos>,    w3},
//     {<eos>,    w4, <eos>, <eos>},
//     {<eos>, <eos>, <eos>, <eos>},
//   }
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

// Helper to save trainer status.
void save_trainer(const string &path, Adam &trainer) {
  ofstream ofs;
  ::open_file(path, ofs);
  ofs << trainer.get_epoch() << endl;
  ofs << trainer.get_weight_decay() << endl;
  ofs << trainer.get_gradient_clipping() << endl;
  ofs << trainer.alpha() << endl;
  ofs << trainer.beta1() << endl;
  ofs << trainer.beta2() << endl;
  ofs << trainer.eps() << endl;
}

// Helper to load trainer status.
Adam load_trainer(const string &path) {
  ifstream ifs;
  ::open_file(path, ifs);
  unsigned epoch;
  float wd, gc, a, b1, b2, eps;
  ifs >> epoch >> wd >> gc >> a >> b1 >> b2 >> eps;
  Adam trainer(a, b1, b2, eps);
  trainer.set_epoch(epoch);
  trainer.set_weight_decay(wd);
  trainer.set_gradient_clipping(gc);
  return trainer;
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
  LSTM(const string &name, unsigned in_size, unsigned out_size)
    : name_(name)
    , out_size_(out_size)
    , pwxh_(name_ + "_wxh", {4 * out_size, in_size}, I::XavierUniform())
    , pwhh_(name_ + "_whh", {4 * out_size, out_size}, I::XavierUniform())
    , pbh_(name_ + "_bh", {4 * out_size}, I::Constant(0)) {}

  // Loads all parameters.
  LSTM(const string &name, const string &prefix)
    : name_(name)
    , pwxh_(Parameter::load(prefix + name_ + "_wxh.yaml"))
    , pwhh_(Parameter::load(prefix + name_ + "_whh.yaml"))
    , pbh_(Parameter::load(prefix + name_ + "_bh.yaml")) {
      out_size_ = pbh_.shape()[0] / 4;
  }

  // Saves all parameters.
  void save(const string &prefix) const {
    pwxh_.save(prefix + name_ + "_wxh.yaml");
    pwhh_.save(prefix + name_ + "_whh.yaml");
    pbh_.save(prefix + name_ + "_bh.yaml");
  }

  // Adds parameters to the trainer.
  void register_training(Trainer &trainer) {
    trainer.add_parameter(pwxh_);
    trainer.add_parameter(pwhh_);
    trainer.add_parameter(pbh_);
  }

  // Initializes internal values.
  void init(const Node &init_c = Node(), const Node &init_h = Node()) {
    wxh_ = F::input(pwxh_);
    whh_ = F::input(pwhh_);
    bh_ = F::input(pbh_);
    c_ = init_c.valid() ? init_c : F::zeros({out_size_});
    h_ = init_h.valid() ? init_h : F::zeros({out_size_});
  }

  // One step forwarding.
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

  // Retrieves current states.
  Node get_c() const { return c_; }
  Node get_h() const { return h_; }

private:
  string name_;
  unsigned out_size_;
  Parameter pwxh_, pwhh_, pbh_;
  Node wxh_, whh_, bh_, h_, c_;
};

// Encoder-decoder translation model.
class EncoderDecoder {
public:
  EncoderDecoder(const string &name,
      unsigned src_vocab_size, unsigned trg_vocab_size,
      unsigned embed_size, unsigned hidden_size, float dropout_rate)
    : name_(name)
    , dropout_rate_(dropout_rate)
    , psrc_lookup_(name_ + "_src_lookup", {embed_size, src_vocab_size}, I::XavierUniform())
    , ptrg_lookup_(name_ + "_trg_lookup", {embed_size, trg_vocab_size}, I::XavierUniform())
    , pwhy_(name_ + "_why", {trg_vocab_size, hidden_size}, I::XavierUniform())
    , pby_(name_ + "_by", {trg_vocab_size}, I::Constant(0))
    , src_lstm_(name_ + "_src_lstm", embed_size, hidden_size)
    , trg_lstm_(name + "_trg_lstm", embed_size, hidden_size) {}

  // Loads all parameters.
  EncoderDecoder(const string &name, const string &prefix)
    : name_(name)
    , psrc_lookup_(Parameter::load(prefix + name_ + "_src_lookup.yaml"))
    , ptrg_lookup_(Parameter::load(prefix + name_ + "_trg_lookup.yaml"))
    , pwhy_(Parameter::load(prefix + name_ + "_why.yaml"))
    , pby_(Parameter::load(prefix + name_ + "_by.yaml"))
    , src_lstm_(name_ + "_src_lstm", prefix)
    , trg_lstm_(name_ + "_trg_lstm", prefix) {
      ifstream ifs;
      ::open_file(prefix + name_ + "_config.txt", ifs);
      ifs >> dropout_rate_;
    }

  // Saves all parameters.
  void save(const string &prefix) const {
    psrc_lookup_.save(prefix + name_ + "_src_lookup.yaml");
    ptrg_lookup_.save(prefix + name_ + "_trg_lookup.yaml");
    pwhy_.save(prefix + name_ + "_why.yaml");
    pby_.save(prefix + name_ + "_by.yaml");
    src_lstm_.save(prefix);
    trg_lstm_.save(prefix);
    ofstream ofs;
    ::open_file(prefix + name_ + "_config.txt", ofs);
    ofs << dropout_rate_ << endl;
  }

  // Adds parameters to the trainer.
  void register_training(Trainer &trainer) {
    trainer.add_parameter(psrc_lookup_);
    trainer.add_parameter(ptrg_lookup_);
    trainer.add_parameter(pwhy_);
    trainer.add_parameter(pby_);
    src_lstm_.register_training(trainer);
    trg_lstm_.register_training(trainer);
  }

  // Encodes source sentences and prepare internal states.
  void encode(const vector<vector<unsigned>> &src_sents, bool train) {
    // Reversed encoding.
    Node src_lookup = F::input(psrc_lookup_);
    src_lstm_.init();
    for (unsigned i = src_sents.size(); i > 0; --i) {
      Node x = F::pick(src_lookup, 1, src_sents[i - 1]);
      x = F::dropout(x, dropout_rate_, train);
      src_lstm_.forward(x);
    }

    // Initializes decoder states.
    trg_lookup_ = F::input(ptrg_lookup_);
    why_ = F::input(pwhy_);
    by_ = F::input(pby_);
    trg_lstm_.init(src_lstm_.get_c(), src_lstm_.get_h());
  }

  // One step decoding.
  Node decode_step(const vector<unsigned> &trg_words, bool train) {
    Node x = F::pick(trg_lookup_, 1, trg_words);
    x = F::dropout(x, dropout_rate_, train);
    Node h = trg_lstm_.forward(x);
    h = F::dropout(h, dropout_rate_, train);
    return F::matmul(why_, h) + by_;
  }

  // Calculates the loss function over given target sentences.
  Node loss(const vector<vector<unsigned>> &trg_sents, bool train) {
    vector<Node> losses;
    for (unsigned i = 0; i < trg_sents.size() - 1; ++i) {
      Node y = decode_step(trg_sents[i], train);
      losses.emplace_back(F::softmax_cross_entropy(y, 0, trg_sents[i + 1]));
    }
    return F::batch::mean(F::sum(losses));
  }

private:
  string name_;
  float dropout_rate_;
  Parameter psrc_lookup_, ptrg_lookup_, pwhy_, pby_;
  ::LSTM src_lstm_, trg_lstm_;
  Node trg_lookup_, why_, by_;
};

}  // namespace

int main(const int argc, const char *argv[]) {
  if (argc != 3) {
    cerr << "usage: " << argv[0] << " (train|resume) model_prefix" << endl;
    exit(1);
  }

  const string mode = argv[1];
  const string prefix = argv[2];
  if (mode != "train" && mode != "resume") {
    cerr << "unknown mode: " << mode << endl;
    exit(1);
  }

  // Loads vocab.
  const auto src_vocab = ::make_vocab("data/train.en", 4000);
  const auto trg_vocab = ::make_vocab("data/train.ja", 5000);
  cout << "#src_vocab: " << src_vocab.size() << endl;
  cout << "#trg_vocab: " << trg_vocab.size() << endl;
  const unsigned src_eos_id = src_vocab.at("<eos>");
  const unsigned trg_eos_id = trg_vocab.at("<eos>");

  // Loads all corpus.
  const auto train_src_corpus = ::load_corpus("data/train.en", src_vocab);
  const auto train_trg_corpus = ::load_corpus("data/train.ja", trg_vocab);
  const auto valid_src_corpus = ::load_corpus("data/dev.en", src_vocab);
  const auto valid_trg_corpus = ::load_corpus("data/dev.ja", trg_vocab);
  const unsigned num_train_sents = train_trg_corpus.size();
  const unsigned num_valid_sents = valid_trg_corpus.size();
  const unsigned num_train_labels = ::count_labels(train_trg_corpus);
  const unsigned num_valid_labels = ::count_labels(valid_trg_corpus);
  cout << "train: " << num_train_sents << " sentences, "
                    << num_train_labels << " labels" << endl;
  cout << "valid: " << num_valid_sents << " sentences, "
                    << num_valid_labels << " labels" << endl;

  // Uses GPU.
  CUDADevice dev(0);
  Device::set_default_device(dev);

  // Trainer.
  unique_ptr<Adam> trainer;
  if (mode == "train") {
    trainer.reset(new Adam());
    trainer->set_weight_decay(1e-6);
    trainer->set_gradient_clipping(5);
    cout << "Created a new trainer: " << prefix << endl;
  } else if (mode == "resume") {
    trainer.reset(new Adam(::load_trainer(prefix + ".trainer.txt")));
    cout << "Loaded an existing trainer: " << prefix << endl;
  }

  // Our translation model.
  unique_ptr<EncoderDecoder> encdec;
  if (mode == "train") {
    encdec.reset(new EncoderDecoder("encdec",
          src_vocab.size(), trg_vocab.size(),
          NUM_EMBED_UNITS, NUM_HIDDEN_UNITS, DROPOUT_RATE));
    cout << "Created a new EncoderDecoder model: " << prefix << endl;
  } else if (mode == "resume") {
    encdec.reset(new EncoderDecoder("encdec", prefix + '.'));
    cout << "Loaded an existing EncoderDecoder model: " << prefix << endl;
  }
  encdec->register_training(*trainer);

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
      const auto src_batch = ::make_batch(
          train_src_corpus, batch_ids, src_eos_id);
      const auto trg_batch = ::make_batch(
          train_trg_corpus, batch_ids, trg_eos_id);
      trainer->reset_gradients();
      Graph g;
      Graph::set_default_graph(g);
      encdec->encode(src_batch, true);
      const auto loss = encdec->loss(trg_batch, true);
      train_loss += g.forward(loss).to_vector()[0] * batch_ids.size();
      g.backward(loss);
      trainer->update(1);
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
      const auto src_batch = ::make_batch(
          valid_src_corpus, batch_ids, src_eos_id);
      const auto trg_batch = ::make_batch(
          valid_trg_corpus, batch_ids, trg_eos_id);
      Graph g;
      Graph::set_default_graph(g);
      encdec->encode(src_batch, false);
      const auto loss = encdec->loss(trg_batch, false);
      valid_loss += g.forward(loss).to_vector()[0] * batch_ids.size();
      cout << ofs << '\r' << flush;
    }
    const float valid_ppl = std::exp(valid_loss / num_valid_labels);
    cout << "  valid ppl = " << valid_ppl << endl;

    encdec->save(prefix + '.');
    ::save_trainer(prefix + ".trainer.txt", *trainer);
    cout << "  saved parameters." << endl;
  }

  return 0;
}
