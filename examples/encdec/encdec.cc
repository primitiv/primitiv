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
//   $ g++
//     -std=c++11
//     -I/path/to/primitiv/includes (typically -I../..)
//     -L/path/to/primitiv/libs     (typically -L../../build/primitiv)
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
#include <random>

#include <primitiv/primitiv.h>

#include "lstm.h"
#include "utils.h"

using namespace primitiv;
using namespace std;
namespace F = primitiv::functions;

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

// Encoder-decoder translation model.
template<typename Var>
class EncoderDecoder : public Model {
  float dropout_rate_;
  Parameter psrc_lookup_, ptrg_lookup_, pwhy_, pby_;
  ::LSTM<Var> src_lstm_, trg_lstm_;
  Var trg_lookup_, why_, by_;

public:
  EncoderDecoder() : dropout_rate_(DROPOUT_RATE) {
    add("src_lookup", psrc_lookup_);
    add("trg_lookup", ptrg_lookup_);
    add("why", pwhy_);
    add("by", pby_);
    add("src_lstm", src_lstm_);
    add("trg_lstm", trg_lstm_);
  }

  // Initializes the model.
  void init(
      unsigned src_vocab_size, unsigned trg_vocab_size,
      unsigned embed_size, unsigned hidden_size) {
    using initializers::XavierUniform;
    using initializers::Constant;
    psrc_lookup_.init({embed_size, src_vocab_size}, XavierUniform());
    ptrg_lookup_.init({embed_size, trg_vocab_size}, XavierUniform());
    pwhy_.init({trg_vocab_size, hidden_size}, XavierUniform());
    pby_.init({trg_vocab_size}, Constant(0));
    src_lstm_.init(embed_size, hidden_size);
    trg_lstm_.init(embed_size, hidden_size);
  }

  // Encodes source sentences and prepare internal states.
  void encode(const vector<vector<unsigned>> &src_batch, bool train) {
    // Reversed encoding.
    Var src_lookup = F::parameter<Var>(psrc_lookup_);
    src_lstm_.restart();
    for (auto it = src_batch.rbegin(); it != src_batch.rend(); ++it) {
      Var x = F::pick(src_lookup, *it, 1);
      x = F::dropout(x, dropout_rate_, train);
      src_lstm_.forward(x);
    }

    // Initializes decoder states.
    trg_lookup_ = F::parameter<Var>(ptrg_lookup_);
    why_ = F::parameter<Var>(pwhy_);
    by_ = F::parameter<Var>(pby_);
    trg_lstm_.restart(src_lstm_.get_c(), src_lstm_.get_h());
  }

  // One step decoding.
  Var decode_step(const vector<unsigned> &trg_words, bool train) {
    Var x = F::pick(trg_lookup_, trg_words, 1);
    x = F::dropout(x, dropout_rate_, train);
    Var h = trg_lstm_.forward(x);
    h = F::dropout(h, dropout_rate_, train);
    return F::matmul(why_, h) + by_;
  }

  // Calculates the loss function over given target sentences.
  Var loss(const vector<vector<unsigned>> &trg_batch, bool train) {
    vector<Var> losses;
    for (unsigned i = 0; i < trg_batch.size() - 1; ++i) {
      Var y = decode_step(trg_batch[i], train);
      losses.emplace_back(F::softmax_cross_entropy(y, trg_batch[i + 1], 0));
    }
    return F::batch::mean(F::sum(losses));
  }
};

// Training encoder decoder model.
void train(
    ::EncoderDecoder<Node> &encdec, Optimizer &optimizer,
    const string &prefix, float best_valid_ppl) {
  // Registers all parameters to the optimizer.
  optimizer.add(encdec);

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

  // Computation graph.
  Graph g;
  Graph::set_default(g);

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

      g.clear();
      encdec.encode(src_batch, true);
      const auto loss = encdec.loss(trg_batch, true);
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
          begin(valid_ids) + std::min<unsigned>(ofs + BATCH_SIZE, num_valid_sents));
      const auto src_batch = ::make_batch(valid_src_corpus, batch_ids, src_vocab);
      const auto trg_batch = ::make_batch(valid_trg_corpus, batch_ids, trg_vocab);

      g.clear();
      encdec.encode(src_batch, false);
      const auto loss = encdec.loss(trg_batch, false);
      valid_loss += loss.to_float() * batch_ids.size();

      cout << ofs << '\r' << flush;
    }

    const float valid_ppl = std::exp(valid_loss / num_valid_labels);
    cout << "  valid ppl = " << valid_ppl << endl;

    // Saves best model/optimizer.
    if (valid_ppl < best_valid_ppl) {
      best_valid_ppl = valid_ppl;
      cout << "  saving model/optimizer ... " << flush;
      encdec.save(prefix + ".model");
      optimizer.save(prefix + ".optimizer");
      ::save_ppl(prefix + ".valid_ppl", best_valid_ppl);
      cout << "done." << endl;
    }
  }
}

// Generates translation by consuming stdin.
void test(::EncoderDecoder<Tensor> &encdec) {
  // Loads vocab.
  const auto src_vocab = ::make_vocab(SRC_TRAIN_FILE, SRC_VOCAB_SIZE);
  const auto trg_vocab = ::make_vocab(TRG_TRAIN_FILE, TRG_VOCAB_SIZE);
  const auto inv_trg_vocab = ::make_inv_vocab(trg_vocab);

  string line;
  while (getline(cin, line)) {
    const vector<vector<unsigned>> src_corpus {::line_to_sent(line, src_vocab)};
    const auto src_batch = ::make_batch(src_corpus, {0}, src_vocab);
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
      trg_ids.emplace_back(y.argmax(0)[0]);
    }

    // Prints the result.
    for (unsigned i = 1; i < trg_ids.size() - 1; ++i) {
      if (i > 1) cout << ' ';
      cout << inv_trg_vocab[trg_ids[i]];
    }
    cout << endl;
  }
}

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
  devices::Naive dev;  // devices::CUDA dev(0);
  Device::set_default(dev);
  cerr << "done." << endl;

  if (mode == "train") {
    ::EncoderDecoder<Node> encdec;
    encdec.init(
        SRC_VOCAB_SIZE, TRG_VOCAB_SIZE, NUM_EMBED_UNITS, NUM_HIDDEN_UNITS);
    optimizers::Adam optimizer;
    optimizer.set_weight_decay(1e-6);
    optimizer.set_gradient_clipping(5);
    ::train(encdec, optimizer, prefix, 1e10);
  } else if (mode == "resume") {
    cerr << "loading model/optimizer ... " << flush;
    ::EncoderDecoder<Node> encdec;
    encdec.load(prefix + ".model");
    optimizers::Adam optimizer;
    optimizer.load(prefix + ".optimizer");
    float valid_ppl = ::load_ppl(prefix + ".valid_ppl");
    cerr << "done." << endl;
    ::train(encdec, optimizer, prefix, valid_ppl);
  } else {  // mode == "test"
    cerr << "loading model ... ";
    ::EncoderDecoder<Tensor> encdec;
    encdec.load(prefix + ".model");
    cerr << "done." << endl;
    ::test(encdec);
  }
  return 0;
}
