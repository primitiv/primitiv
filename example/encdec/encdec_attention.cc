// Sample code to train the encoder-decoder model with dot-attention mechanism.
//
// Model details:
//   Bidirectional encoder and basics of attention mechanism:
//     Bahdanau et al., 2015.
//     Neural Machine Translation by Jointly Learning to Align and Translate.
//     https://arxiv.org/abs/1409.0473
//   Decoder design and dot-attention:
//     Luong et al., 2015.
//     Effective Approaches to Attention-based Neural Machine Translation
//     https://arxiv.org/abs/1508.04025
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
//     encdec_attention.cc -lprimitiv
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
#include <primitiv/primitiv_cuda.h>

#include "lstm.h"
#include "utils.h"

using namespace primitiv;
using namespace std;
namespace F = primitiv::operators;
namespace I = primitiv::initializers;

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

// Encoder-decoder translation model with dot-attention.
template<typename Var>
class EncoderDecoder {
  string name_;
  unsigned embed_size_;
  float dropout_rate_;
  Parameter psrc_lookup_, ptrg_lookup_, pwhj_, pbj_, pwjy_, pby_;
  ::LSTM<Var> src_fw_lstm_, src_bw_lstm_, trg_lstm_;
  Var trg_lookup_, whj_, bj_, wjy_, by_, concat_fb_, t_concat_fb_, feed_;

public:
  EncoderDecoder(const string &name,
      unsigned src_vocab_size, unsigned trg_vocab_size,
      unsigned embed_size, unsigned hidden_size, float dropout_rate)
    : name_(name)
    , embed_size_(embed_size)
    , dropout_rate_(dropout_rate)
    , psrc_lookup_(name_ + "_src_lookup", {embed_size, src_vocab_size}, I::XavierUniform())
    , ptrg_lookup_(name_ + "_trg_lookup", {embed_size, trg_vocab_size}, I::XavierUniform())
    , pwhj_(name_ + "_whj", {embed_size, 2 * hidden_size}, I::XavierUniform())
    , pbj_(name_ + "_bj", {embed_size}, I::Constant(0))
    , pwjy_(name_ + "_wjy", {trg_vocab_size, embed_size}, I::XavierUniform())
    , pby_(name_ + "_by", {trg_vocab_size}, I::Constant(0))
    , src_fw_lstm_(name_ + "_src_fw_lstm", embed_size, hidden_size)
    , src_bw_lstm_(name_ + "_src_bw_lstm", embed_size, hidden_size)
    , trg_lstm_(name + "_trg_lstm", 2 * embed_size, hidden_size) {}

  // Loads all parameters.
  EncoderDecoder(const string &name, const string &prefix)
    : name_(name)
    , psrc_lookup_(Parameter::load(prefix + name_ + "_src_lookup.param"))
    , ptrg_lookup_(Parameter::load(prefix + name_ + "_trg_lookup.param"))
    , pwhj_(Parameter::load(prefix + name_ + "_whj.param"))
    , pbj_(Parameter::load(prefix + name_ + "_bj.param"))
    , pwjy_(Parameter::load(prefix + name_ + "_wjy.param"))
    , pby_(Parameter::load(prefix + name_ + "_by.param"))
    , src_fw_lstm_(name_ + "_src_fw_lstm", prefix)
    , src_bw_lstm_(name_ + "_src_bw_lstm", prefix)
    , trg_lstm_(name_ + "_trg_lstm", prefix) {
      embed_size_ = pbj_.shape()[0];
      ifstream ifs;
      ::open_file(prefix + name_ + "_config.config", ifs);
      ifs >> dropout_rate_;
    }

  // Saves all parameters.
  void save(const string &prefix) const {
    psrc_lookup_.save(prefix + name_ + "_src_lookup.param");
    ptrg_lookup_.save(prefix + name_ + "_trg_lookup.param");
    pwhj_.save(prefix + name_ + "_whj.param");
    pbj_.save(prefix + name_ + "_bj.param");
    pwjy_.save(prefix + name_ + "_wjy.param");
    pby_.save(prefix + name_ + "_by.param");
    src_fw_lstm_.save(prefix);
    src_bw_lstm_.save(prefix);
    trg_lstm_.save(prefix);
    ofstream ofs;
    ::open_file(prefix + name_ + "_config.config", ofs);
    ofs << dropout_rate_ << endl;
  }

  // Adds parameters to the trainer.
  void register_training(Trainer &trainer) {
    trainer.add_parameter(psrc_lookup_);
    trainer.add_parameter(ptrg_lookup_);
    trainer.add_parameter(pwhj_);
    trainer.add_parameter(pbj_);
    trainer.add_parameter(pwjy_);
    trainer.add_parameter(pby_);
    src_fw_lstm_.register_training(trainer);
    src_bw_lstm_.register_training(trainer);
    trg_lstm_.register_training(trainer);
  }

  // Encodes source sentences and prepare internal states.
  void encode(const vector<vector<unsigned>> &src_batch, bool train) {
    // Embedding lookup.
    const Var src_lookup = F::parameter<Var>(psrc_lookup_);
    vector<Var> e_list;
    for (const auto &x : src_batch) {
      e_list.emplace_back(
          F::dropout(F::pick(src_lookup, x, 1), dropout_rate_, train));
    }

    // Forward encoding.
    src_fw_lstm_.init();
    vector<Var> f_list;
    for (const auto &e : e_list) {
      f_list.emplace_back(
          F::dropout(src_fw_lstm_.forward(e), dropout_rate_, train));
    }

    // Backward encoding.
    src_bw_lstm_.init();
    vector<Var> b_list;
    for (auto it = e_list.rbegin(); it != e_list.rend(); ++it) {
      b_list.emplace_back(
          F::dropout(src_bw_lstm_.forward(*it), dropout_rate_, train));
    }
    reverse(begin(b_list), end(b_list));

    // Concatenates RNN states.
    vector<Var> fb_list;
    for (unsigned i = 0; i < src_batch.size(); ++i) {
      fb_list.emplace_back(f_list[i] + b_list[i]);
    }
    concat_fb_ = F::concat(fb_list, 1);
    t_concat_fb_ = F::transpose(concat_fb_);

    // Initializes decoder states.
    trg_lookup_ = F::parameter<Var>(ptrg_lookup_);
    whj_ = F::parameter<Var>(pwhj_);
    bj_ = F::parameter<Var>(pbj_);
    wjy_ = F::parameter<Var>(pwjy_);
    by_ = F::parameter<Var>(pby_);
    feed_ = F::zeros<Var>({embed_size_});
    trg_lstm_.init(
        src_fw_lstm_.get_c() + src_bw_lstm_.get_c(),
        src_fw_lstm_.get_h() + src_bw_lstm_.get_h());
  }

  // One step decoding.
  Var decode_step(const vector<unsigned> &trg_words, bool train) {
    Var e = F::pick(trg_lookup_, trg_words, 1);
    e = F::dropout(e, dropout_rate_, train);
    Var h = trg_lstm_.forward(F::concat({e, feed_}, 0));
    h = F::dropout(h, dropout_rate_, train);
    const Var atten_probs = F::softmax(F::matmul(t_concat_fb_, h), 0);
    const Var c = F::matmul(concat_fb_, atten_probs);
    feed_ = F::tanh(F::matmul(whj_, F::concat({h, c}, 0)) + bj_);
    return F::matmul(wjy_, feed_) + by_;
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
    EncoderDecoder<Node> &encdec, Trainer &trainer, const string &prefix,
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
    cout << "epoch " << (epoch + 1) << '/' << MAX_EPOCH
         << ", lr_scale = " << trainer.get_learning_rate_scaling() << endl;
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
      DefaultScope<Graph> gs(g);
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
      DefaultScope<Graph> gs(g);
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
    } else {
      // Learning rate decay by 1/sqrt(2)
      const float new_scale = .7071 * trainer.get_learning_rate_scaling();
      trainer.set_learning_rate_scaling(new_scale);
    }
  }
}

// Generates translation by consuming stdin.
void test(EncoderDecoder<Tensor> &encdec) {
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
      const auto logits = y.to_vector();
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
  DefaultScope<Device> ds(dev);
  cerr << "done." << endl;

  if (mode == "train") {
    ::EncoderDecoder<Node> encdec("encdec",
        SRC_VOCAB_SIZE, TRG_VOCAB_SIZE,
        NUM_EMBED_UNITS, NUM_HIDDEN_UNITS, DROPOUT_RATE);
    trainers::Adam trainer;
    trainer.set_weight_decay(1e-6);
    trainer.set_gradient_clipping(5);
    ::train(encdec, trainer, prefix, 1e10);
  } else if (mode == "resume") {
    cerr << "loading model/trainer ... " << flush;
    ::EncoderDecoder<Node> encdec("encdec", prefix + '.');
    shared_ptr<Trainer> trainer = Trainer::load(prefix + ".trainer.config");
    float valid_ppl = ::load_ppl(prefix + ".valid_ppl.config");
    cerr << "done." << endl;
    ::train(encdec, *trainer, prefix, valid_ppl);
  } else {  // mode == "test"
    cerr << "loading model ... ";
    ::EncoderDecoder<Tensor> encdec("encdec", prefix + '.');
    cerr << "done." << endl;
    ::test(encdec);
  }
  return 0;
}
