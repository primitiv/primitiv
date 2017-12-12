#ifndef PRIMITIV_EXAMPLES_PTB_UTILS_H_
#define PRIMITIV_EXAMPLES_PTB_UTILS_H_

/**
 * Common utility functions for PTB examples.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace utils {

// Gathers the set of words from space-separated corpus.
std::unordered_map<std::string, unsigned> make_vocab(
    const std::string &filename) {
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    std::cerr << "File could not be opened: " << filename << std::endl;
    exit(1);
  }
  std::unordered_map<std::string, unsigned> vocab;
  std::string line, word;
  while (getline(ifs, line)) {
    line = "<s>" + line + "<s>";
    std::stringstream ss(line);
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
std::vector<std::vector<unsigned>> load_corpus(
    const std::string &filename,
    const std::unordered_map<std::string, unsigned> &vocab) {
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    std::cerr << "File could not be opened: " << filename << std::endl;
    exit(1);
  }
  std::vector<std::vector<unsigned>> corpus;
  std::string line, word;
  while (getline(ifs, line)) {
    line = "<s>" + line + "<s>";
    std::stringstream ss (line);
    std::vector<unsigned> sentence;
    while (getline(ss, word, ' ')) {
      sentence.emplace_back(vocab.at(word));
    }
    corpus.emplace_back(move(sentence));
  }
  return corpus;
}

// Counts output labels in the corpus.
unsigned count_labels(const std::vector<std::vector<unsigned>> &corpus) {
  unsigned ret = 0;
  for (const auto &sent :corpus) ret += sent.size() - 1;
  return ret;
}

// Extracts a minibatch from loaded corpus
std::vector<std::vector<unsigned>> make_batch(
    const std::vector<std::vector<unsigned>> &corpus,
    const std::vector<unsigned> &sent_ids,
    unsigned eos_id) {
  const unsigned batch_size = sent_ids.size();
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

}  // namespace utils

#endif  // PRIMITIV_EXAMPLES_PTB_UTILS_H_
