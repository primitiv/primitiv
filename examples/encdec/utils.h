#ifndef PRIMITIV_EXAMPLE_ENCDEC_UTILS_H_
#define PRIMITIV_EXAMPLE_ENCDEC_UTILS_H_

#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

// Helper to open fstream
template <class FStreamT>
inline void open_file(const std::string &path, FStreamT &fs) {
  fs.open(path);
  if (!fs.is_open()) {
    std::cerr << "File could not be opened: " + path << std::endl;
    exit(1);
  }
}

// Gathers the set of words from space-separated corpus and makes a vocabulary.
inline std::unordered_map<std::string, unsigned> make_vocab(
    const std::string &path, unsigned size) {
  if (size < 3) {
    std::cerr << "Vocab size should be <= 3." << std::endl;
    exit(1);
  }
  std::ifstream ifs;
  ::open_file(path, ifs);

  // Counts all word existences.
  std::unordered_map<std::string, unsigned> freq;
  std::string line, word;
  while (getline(ifs, line)) {
    std::stringstream ss(line);
    while (getline(ss, word, ' ')) ++freq[word];
  }

  // Sorting.
  using freq_t = std::pair<std::string, unsigned>;
  auto cmp = [](const freq_t &a, const freq_t &b) {
    return a.second < b.second;
  };
  std::priority_queue<freq_t, std::vector<freq_t>, decltype(cmp)> q(cmp);
  for (const auto &x : freq) q.push(x);

  // Chooses top size-3 frequent words to make the vocabulary.
  std::unordered_map<std::string, unsigned> vocab;
  vocab.emplace("<unk>", 0);
  vocab.emplace("<bos>", 1);
  vocab.emplace("<eos>", 2);
  for (unsigned i = 3; i < size; ++i) {
    vocab.emplace(q.top().first, i);
    q.pop();
  }
  return vocab;
}

// Generates ID-to-word dictionary.
inline std::vector<std::string> make_inv_vocab(
    const std::unordered_map<std::string, unsigned> &vocab) {
  std::vector<std::string> ret(vocab.size());
  for (const auto &kv : vocab) {
    ret[kv.second] = kv.first;
  }
  return ret;
}

// Generates word ID list from a sentence.
inline std::vector<unsigned> line_to_sent(
    const std::string &line,
    const std::unordered_map<std::string, unsigned> &vocab) {
  const unsigned unk_id = vocab.at("<unk>");
  std::string converted = "<bos> " + line + " <eos>";
  std::stringstream ss(converted);
  std::vector<unsigned> sent;
  std::string word;
  while (getline(ss, word, ' ')) {
    const auto it = vocab.find(word);
    if (it != vocab.end()) sent.emplace_back(it->second);
    else sent.emplace_back(unk_id);
  }
  return sent;
}

// Generates word ID list from a corpus.
// All out-of-vocab words are replaced to <unk>.
inline std::vector<std::vector<unsigned>> load_corpus(
    const std::string &path,
    const std::unordered_map<std::string, unsigned> &vocab) {
  std::ifstream ifs;
  ::open_file(path, ifs);
  std::vector<std::vector<unsigned>> corpus;
  std::string line, word;
  while (getline(ifs, line)) corpus.emplace_back(::line_to_sent(line, vocab));
  return corpus;
}

// Counts output labels in the corpus.
inline unsigned count_labels(const std::vector<std::vector<unsigned>> &corpus) {
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
inline std::vector<std::vector<unsigned>> make_batch(
    const std::vector<std::vector<unsigned>> &corpus,
    const std::vector<unsigned> &sent_ids,
    const std::unordered_map<std::string, unsigned> &vocab) {
  const unsigned batch_size = sent_ids.size();
  const unsigned eos_id = vocab.at("<eos>");
  unsigned max_len = 0;
  for (const unsigned sid : sent_ids) {
    max_len = std::max<unsigned>(max_len, corpus[sid].size());
  }
  std::vector<std::vector<unsigned>> batch(
      max_len, std::vector<unsigned>(batch_size, eos_id));
  for (unsigned i = 0; i < batch_size; ++i) {
    const auto &sent = corpus[sent_ids[i]];
    for (unsigned j = 0; j < sent.size(); ++j) {
      batch[j][i] = sent[j];
    }
  }
  return batch;
}

// Helper to save current ppl.
inline void save_ppl(const std::string &path, float ppl) {
  std::ofstream ofs;
  ::open_file(path, ofs);
  ofs << ppl << std::endl;
}

// Helper to load last ppl.
inline float load_ppl(const std::string &path) {
  std::ifstream ifs;
  ::open_file(path, ifs);
  float ppl;
  ifs >> ppl;
  return ppl;
}

#endif  // PRIMITIV_EXAMPLE_ENCDEC_UTILS_H_
