#ifndef PRIMITIV_EXAMPLES_MNIST_UTILS_H_
#define PRIMITIV_EXAMPLES_MNIST_UTILS_H_

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace utils {

// Helper function to load input images.
inline std::vector<float> load_mnist_images(
    const std::string &filename, const unsigned n) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "File could not be opened: " << filename << std::endl;
    std::abort();
  }

  ifs.ignore(16);  // header
  const unsigned size = n * 28 * 28;
  std::vector<unsigned char> buf(size);
  ifs.read(reinterpret_cast<char *>(&buf[0]), size);
  std::vector<float> ret(size);
  for (unsigned i = 0; i < size; ++i) ret[i] = buf[i] / 255.0;
  return ret;
}

// Helper function to load labels.
inline std::vector<char> load_mnist_labels(
    const std::string &filename, const unsigned n) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "File could not be opened: " << filename << std::endl;
    std::abort();
  }

  ifs.ignore(8);  // header
  std::vector<char> ret(n);
  ifs.read(&ret[0], n);
  return ret;
}

}  // namespace utils

#endif  // PRIMITIV_EXAMPLES_MNIST_UTILS_H_
