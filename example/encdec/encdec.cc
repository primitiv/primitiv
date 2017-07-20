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
#include <random>
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

// Gathers the set of words from space-separated corpus.
unordered_map<string, unsigned> make_vocab(const string &filename) {
  ifstream ifs(filename);
  if (!ifs.is_open()) {
    cerr << "File could not be opened: " << filename << endl;
    exit(1);
  }
  unordered_map<string, unsigned> vocab;
  vocab.emplace(make_pair("<unk>", 0));
  string line, word;
  while (getline(ifs, line)) {
    line = "<s> " + line + " <s>";
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
    line = "<s> " + line + " <s>";
    stringstream ss (line);
    vector<unsigned> sentence;
    while (getline(ss, word, ' ')) {
      const auto it = vocab.find(word);
      if (it != vocab.end()) sentence.emplace_back(it->second);
      else sentence.emplace_back(0);  // <unk>
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
    LSTM(const string &name,
        unsigned in_size, unsigned out_size, Trainer &trainer)
      : out_size_(out_size)
      , pwxh_(name + "_wxh", {4 * out_size, in_size}, I::XavierUniform())
      , pwhh_(name + "_whh", {4 * out_size, out_size}, I::XavierUniform())
      , pbh_(name + "_bh", {4 * out_size}, I::Constant(0)) {
        trainer.add_parameter(pwxh_);
        trainer.add_parameter(pwhh_);
        trainer.add_parameter(pbh_);
      }

    // Initializes internal values.
    void init(const Node &init_c = Node()) {
      wxh_ = F::input(pwxh_);
      whh_ = F::input(pwhh_);
      bh_ = F::input(pbh_);
      if (!init_c.valid()) {
        h_ = c_ = F::zeros({out_size_});
      } else {
        c_ = init_c;
        h_ = F::tanh(c_);
      }
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

    // Retrieves current cell.
    Node get_cell() const { return c_; }

  private:
    unsigned out_size_;
    Parameter pwxh_, pwhh_, pbh_;
    Node wxh_, whh_, bh_, h_, c_;
};

// Encoder-decoder translation model.
class EncoderDecoder {
public:
  EncoderDecoder(unsigned src_vocab_size, unsigned trg_vocab_size,
      unsigned embed_size, unsigned hidden_size, Trainer &trainer)
    : psrc_lookup_(
        "src_lookup", {embed_size, src_vocab_size}, I::XavierUniform())
    , ptrg_lookup_(
        "trg_lookup", {embed_size, trg_vocab_size}, I::XavierUniform())
    , pwhy_("why", {trg_vocab_size, hidden_size}, I::XavierUniform())
    , pby_("by", {trg_vocab_size}, I::Constant(0))
    , src_lstm_("src_lstm", embed_size, hidden_size, trainer)
    , trg_lstm_("trg_lstm", embed_size, hidden_size, trainer) {
      trainer.add_parameter(psrc_lookup_);
      trainer.add_parameter(ptrg_lookup_);
      trainer.add_parameter(pwhy_);
      trainer.add_parameter(pby_);
    }

  // Forward function for training encoder-decoder. Both source and target data
  // should be arranged as:
  // src_tokens, trg_tokens = {
  //   {sent1_word1, sent2_word1, ..., sentN_word1},  // 1st token (<s>)
  //   {sent1_word2, sent2_word2, ..., sentN_word2},  // 2nd token (first word)
  //   ...,
  //   {sent1_wordM, sent2_wordM, ..., sentN_wordM},  // last token (<s>)
  // };
  Node forward_loss(
      const vector<vector<unsigned>> &src_tokens,
      const vector<vector<unsigned>> &trg_tokens,
      bool train) {
    Node src_lookup = F::input(psrc_lookup_);
    Node trg_lookup = F::input(ptrg_lookup_);
    Node why = F::input(pwhy_);
    Node by = F::input(pby_);

    // Reversed encoding (w/o first <s>)
    src_lstm_.init();
    for (unsigned i = src_tokens.size(); i > 1; --i) {
      Node x = F::pick(src_lookup, 1, src_tokens[i - 1]);
      x = F::dropout(x, DROPOUT_RATE, train);
      src_lstm_.forward(x);
    }

    // Decoding
    vector<Node> losses;
    trg_lstm_.init(src_lstm_.get_cell());
    for (unsigned i = 0; i < trg_tokens.size() - 1; ++i) {
      Node x = F::pick(trg_lookup, 1, trg_tokens[i]);
      x = F::dropout(x, DROPOUT_RATE, train);
      Node h = trg_lstm_.forward(x);
      h = F::dropout(h, DROPOUT_RATE, train);
      Node y = F::matmul(why, h) + by;
      losses.emplace_back(F::softmax_cross_entropy(y, 0, trg_tokens[i + 1]));
    }

    return F::batch::mean(F::sum(losses));
  }

private:
  Parameter psrc_lookup_, ptrg_lookup_, pwhy_, pby_;
  ::LSTM src_lstm_, trg_lstm_;
};

}  // namespace

int main() {
  // Loads vocab.
  const auto src_vocab = ::make_vocab("data/train.en");
  const auto trg_vocab = ::make_vocab("data/train.ja");
  cout << "#src_vocab: " << src_vocab.size() << endl;
  cout << "#trg_vocab: " << trg_vocab.size() << endl;
  const unsigned src_eos_id = src_vocab.at("<s>");
  const unsigned trg_eos_id = trg_vocab.at("<s>");

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
  Adam trainer;
  trainer.set_weight_decay(1e-6);
  trainer.set_gradient_clipping(5);

  // Our translation model.
  ::EncoderDecoder encdec(
      src_vocab.size(), trg_vocab.size(),
      NUM_EMBED_UNITS, NUM_HIDDEN_UNITS, trainer);

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
      trainer.reset_gradients();
      Graph g;
      Graph::set_default_graph(g);
      const auto loss = encdec.forward_loss(src_batch, trg_batch, true);
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
      const auto src_batch = ::make_batch(
          valid_src_corpus, batch_ids, src_eos_id);
      const auto trg_batch = ::make_batch(
          valid_trg_corpus, batch_ids, trg_eos_id);
      Graph g;
      Graph::set_default_graph(g);
      const auto loss = encdec.forward_loss(src_batch, trg_batch, false);
      valid_loss += g.forward(loss).to_vector()[0] * batch_ids.size();
      cout << ofs << '\r' << flush;
    }
    const float valid_ppl = std::exp(valid_loss / num_valid_labels);
    cout << "  valid ppl = " << valid_ppl << endl;
  }

  return 0;
}
