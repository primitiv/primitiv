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
//
// [Compile]
//   $ g++ \
//     -std=c++11 \
//     -I/path/to/primitiv/includes \ (typically -I../..)
//     -L/path/to/primitiv/libs \     (typically -L../../build/primitiv)
//     encdec.cc -lprimitiv
//
// [Training]
//   $ ./a.out train <model_prefix>
//
// [Resuming training]
//   $ ./a.out resume <model_prefix>
//
// [Test]
//   $ ./a.out test <model_prefix> < data/test.en > test.hyp.ja

#include <algorithm>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <primitiv/primitiv.h>
#include <primitiv/primitiv_cuda.h>

#include "lstm.h"

using namespace std;
using namespace primitiv;
namespace F = primitiv::node_ops;
namespace I = primitiv::initializers;

namespace {

static const unsigned SRC_VOCAB_SIZE = 4000;
static const unsigned TRG_VOCAB_SIZE = 5000;
static const unsigned NUM_EMBED_UNITS = 512;
static const unsigned NUM_HIDDEN_UNITS = 512;
static const unsigned BATCH_SIZE = 64;
static const unsigned MAX_EPOCH = 100;
static const float DROPOUT_RATE = 0.5;
static const unsigned GENERATION_LIMIT = 32;

static const char *SRC_TRAIN_FILE = "data/train.en";
static const char *TRG_TRAIN_FILE = "data/train.ja";
static const char *SRC_VALID_FILE = "data/dev.en";
static const char *TRG_VALID_FILE = "data/dev.ja";

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

// Generates ID-to-word dictionary.
vector<string> make_inv_vocab(const unordered_map<string, unsigned> &vocab) {
  vector<string> ret(vocab.size());
  for (const auto &kv : vocab) {
    ret[kv.second] = kv.first;
  }
  return ret;
}

// Generates word ID list from a sentence.
vector<unsigned> line_to_sent(
    const string &line, const unordered_map<string, unsigned> &vocab) {
  const unsigned unk_id = vocab.at("<unk>");
  string converted = "<bos> " + line + " <eos>";
  stringstream ss(converted);
  vector<unsigned> sent;
  string word;
  while (getline(ss, word, ' ')) {
    const auto it = vocab.find(word);
    if (it != vocab.end()) sent.emplace_back(it->second);
    else sent.emplace_back(unk_id);
  }
  return sent;
}

// Generates word ID list from a corpus.
// All out-of-vocab words are replaced to <unk>.
vector<vector<unsigned>> load_corpus(
    const string &path, const unordered_map<string, unsigned> &vocab) {
  const unsigned unk_id = vocab.at("<unk>");
  ifstream ifs;
  ::open_file(path, ifs);
  vector<vector<unsigned>> corpus;
  string line, word;
  while (getline(ifs, line)) corpus.emplace_back(::line_to_sent(line, vocab));
  return corpus;
}

// Counts output labels in the corpus.
unsigned count_labels(const vector<vector<unsigned>> &corpus) {
  unsigned ret = 0;
  for (const auto &sent : corpus) ret += sent.size() - 1;  // w/o <bos>
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
    const unordered_map<string, unsigned> &vocab) {
  const unsigned batch_size = sent_ids.size();
  const unsigned eos_id = vocab.at("<eos>");
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

// Helper to save current ppl.
void save_ppl(const string &path, float ppl) {
  ofstream ofs;
  ::open_file(path, ofs);
  ofs << ppl << endl;
}

// Helper to load last ppl.
float load_ppl(const string &path) {
  ifstream ifs;
  ::open_file(path, ifs);
  float ppl;
  ifs >> ppl;
  return ppl;
}

// Finds a word ID with the highest logit.
unsigned argmax(const vector<float> &logits) {
  unsigned ret = -1;
  float best = -1e10;
  for (unsigned i = 0; i < logits.size(); ++i) {
    if (logits[i] > best) {
      ret = i;
      best = logits[i];
    }
  }
  return ret;
}

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
    , psrc_lookup_(Parameter::load(prefix + name_ + "_src_lookup.param"))
    , ptrg_lookup_(Parameter::load(prefix + name_ + "_trg_lookup.param"))
    , pwhy_(Parameter::load(prefix + name_ + "_why.param"))
    , pby_(Parameter::load(prefix + name_ + "_by.param"))
    , src_lstm_(name_ + "_src_lstm", prefix)
    , trg_lstm_(name_ + "_trg_lstm", prefix) {
      ifstream ifs;
      ::open_file(prefix + name_ + ".config", ifs);
      ifs >> dropout_rate_;
    }

  // Saves all parameters.
  void save(const string &prefix) const {
    psrc_lookup_.save(prefix + name_ + "_src_lookup.param");
    ptrg_lookup_.save(prefix + name_ + "_trg_lookup.param");
    pwhy_.save(prefix + name_ + "_why.param");
    pby_.save(prefix + name_ + "_by.param");
    src_lstm_.save(prefix);
    trg_lstm_.save(prefix);
    ofstream ofs;
    ::open_file(prefix + name_ + ".config", ofs);
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
  void encode(const vector<vector<unsigned>> &src_batch, bool train) {
    // Reversed encoding.
    Node src_lookup = F::input(psrc_lookup_);
    src_lstm_.init();
    for (auto it = src_batch.rbegin(); it != src_batch.rend(); ++it) {
      Node x = F::pick(src_lookup, *it, 1);
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
    Node x = F::pick(trg_lookup_, trg_words, 1);
    x = F::dropout(x, dropout_rate_, train);
    Node h = trg_lstm_.forward(x);
    h = F::dropout(h, dropout_rate_, train);
    return F::matmul(why_, h) + by_;
  }

  // Calculates the loss function over given target sentences.
  Node loss(const vector<vector<unsigned>> &trg_batch, bool train) {
    vector<Node> losses;
    for (unsigned i = 0; i < trg_batch.size() - 1; ++i) {
      Node y = decode_step(trg_batch[i], train);
      losses.emplace_back(F::softmax_cross_entropy(y, trg_batch[i + 1], 0));
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

// Training encoder decoder model.
void train(
    EncoderDecoder &encdec, Trainer &trainer, const string &prefix,
    float best_valid_ppl) {
  // Registers all parameters to the trainer.
  encdec.register_training(trainer);

  // Loads vocab.
  const auto src_vocab = ::make_vocab(SRC_TRAIN_FILE, SRC_VOCAB_SIZE);
  const auto trg_vocab = ::make_vocab(TRG_TRAIN_FILE, TRG_VOCAB_SIZE);
  cout << "#src_vocab: " << src_vocab.size() << endl;  // == SRC_VOCAB_SIZE
  cout << "#trg_vocab: " << trg_vocab.size() << endl;  // == TRG_VOCAB_SIZE

  // Loads all corpus.
  const auto train_src_corpus = ::load_corpus(SRC_TRAIN_FILE, src_vocab);
  const auto train_trg_corpus = ::load_corpus(TRG_TRAIN_FILE, trg_vocab);
  const auto valid_src_corpus = ::load_corpus(SRC_VALID_FILE, src_vocab);
  const auto valid_trg_corpus = ::load_corpus(TRG_VALID_FILE, trg_vocab);
  const unsigned num_train_sents = train_trg_corpus.size();
  const unsigned num_valid_sents = valid_trg_corpus.size();
  const unsigned num_train_labels = ::count_labels(train_trg_corpus);
  const unsigned num_valid_labels = ::count_labels(valid_trg_corpus);
  cout << "train: " << num_train_sents << " sentences, "
                    << num_train_labels << " labels" << endl;
  cout << "valid: " << num_valid_sents << " sentences, "
                    << num_valid_labels << " labels" << endl;

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
          begin(train_ids) + std::min<unsigned>(ofs + BATCH_SIZE, num_train_sents));
      const auto src_batch = ::make_batch(train_src_corpus, batch_ids, src_vocab);
      const auto trg_batch = ::make_batch(train_trg_corpus, batch_ids, trg_vocab);
      trainer.reset_gradients();
      Graph g;
      Graph::set_default_graph(g);
      encdec.encode(src_batch, true);
      const auto loss = encdec.loss(trg_batch, true);
      train_loss += g.forward(loss).to_vector()[0] * batch_ids.size();
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
          begin(valid_ids) + std::min<unsigned>(ofs + BATCH_SIZE, num_valid_sents));
      const auto src_batch = ::make_batch(valid_src_corpus, batch_ids, src_vocab);
      const auto trg_batch = ::make_batch(valid_trg_corpus, batch_ids, trg_vocab);
      Graph g;
      Graph::set_default_graph(g);
      encdec.encode(src_batch, false);
      const auto loss = encdec.loss(trg_batch, false);
      valid_loss += g.forward(loss).to_vector()[0] * batch_ids.size();
      cout << ofs << '\r' << flush;
    }
    const float valid_ppl = std::exp(valid_loss / num_valid_labels);
    cout << "  valid ppl = " << valid_ppl << endl;

    // Saves best model/trainer.
    if (valid_ppl < best_valid_ppl) {
      best_valid_ppl = valid_ppl;
      cout << "  saving model/trainer ... " << flush;
      encdec.save(prefix + '.');
      trainer.save(prefix + ".trainer.config");
      ::save_ppl(prefix + ".valid_ppl.config", best_valid_ppl);
      cout << "done." << endl;
    }
  }
}

// Generates translation by consuming stdin.
void test(EncoderDecoder &encdec) {
  // Loads vocab.
  const auto src_vocab = ::make_vocab(SRC_TRAIN_FILE, SRC_VOCAB_SIZE);
  const auto trg_vocab = ::make_vocab(TRG_TRAIN_FILE, TRG_VOCAB_SIZE);
  const auto inv_trg_vocab = ::make_inv_vocab(trg_vocab);

  string line;
  while (getline(cin, line)) {
    const vector<vector<unsigned>> src_corpus {::line_to_sent(line, src_vocab)};
    const auto src_batch = ::make_batch(src_corpus, {0}, src_vocab);
    Graph g;
    Graph::set_default_graph(g);
    encdec.encode(src_batch, false);

    // Generates target words one-by-one.
    vector<unsigned> trg_ids {trg_vocab.at("<bos>")};
    const unsigned eos_id = trg_vocab.at("<eos>");
    while (trg_ids.back() != eos_id) {
      if (trg_ids.size() > GENERATION_LIMIT + 1) {
        cerr << "Warning: Sentence generation did not finish in "
             << GENERATION_LIMIT << " iterations." << endl;
        trg_ids.emplace_back(eos_id);
        break;
      }
      const auto y = encdec.decode_step({trg_ids.back()}, false);
      const auto logits = g.forward(y).to_vector();
      trg_ids.emplace_back(::argmax(logits));
    }

    // Prints the result.
    for (unsigned i = 1; i < trg_ids.size() - 1; ++i) {
      if (i > 1) cout << ' ';
      cout << inv_trg_vocab[trg_ids[i]];
    }
    cout << endl;
  }
}

}  // namespace

int main(const int argc, const char *argv[]) {
  if (argc != 3) {
    cerr << "usage: " << argv[0]
         << " (train|resume|test) <model_prefix>" << endl;
    exit(1);
  }

  const string mode = argv[1];
  const string prefix = argv[2];
  cerr << "mode: " << mode << endl;
  cerr << "prefix: " << prefix << endl;
  if (mode != "train" && mode != "resume" && mode != "test") {
    cerr << "unknown mode: " << mode << endl;
    exit(1);
  }

  cerr << "initializing device ... " << flush;
  CUDADevice dev(0);
  Device::set_default_device(dev);
  cerr << "done." << endl;

  if (mode == "train") {
    EncoderDecoder encdec("encdec",
        SRC_VOCAB_SIZE, TRG_VOCAB_SIZE,
        NUM_EMBED_UNITS, NUM_HIDDEN_UNITS, DROPOUT_RATE);
    trainers::Adam trainer;
    trainer.set_weight_decay(1e-6);
    trainer.set_gradient_clipping(5);
    ::train(encdec, trainer, prefix, 1e10);
  } else if (mode == "resume") {
    cerr << "loading model/trainer ... " << flush;
    EncoderDecoder encdec("encdec", prefix + '.');
    shared_ptr<Trainer> trainer = Trainer::load(prefix + ".trainer.config");
    float valid_ppl = ::load_ppl(prefix + ".valid_ppl.config");
    cerr << "done." << endl;
    ::train(encdec, *trainer, prefix, valid_ppl);
  } else {  // mode == "test"
    cerr << "loading model ... ";
    EncoderDecoder encdec("encdec", prefix + '.');
    cerr << "done." << endl;
    ::test(encdec);
  }
  return 0;
}
