#ifndef PRIMITIV_TEST_UTILS_H_
#define PRIMITIV_TEST_UTILS_H_

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include <primitiv/primitiv.h>

namespace test_utils {

// check whether or not two float values are near than the ULP-based threshold.
inline bool float_eq(const float a, const float b) {
  static_assert(sizeof(std::int32_t) == sizeof(float), "");
  static const int MAX_ULPS = 4;
  std::int32_t ai;
  std::memcpy(&ai, &a, sizeof(std::int32_t));
  if (ai < 0) ai = 0x80000000 - ai;
  std::int32_t bi;
  std::memcpy(&bi, &b, sizeof(std::int32_t));
  if (bi < 0) bi = 0x80000000 - bi;
  const std::int32_t diff = ai > bi ? ai - bi : bi - ai;
  return (diff <= MAX_ULPS);
}

// check whether or not two float values are near than the given error.
inline bool float_near(const float a, const float b, const float err) {
  return (a > b ? a - b : b - a) <= err;
}

// helper to check vector equality.
template<typename T>
inline testing::AssertionResult vector_match(
    const std::vector<T> &expected,
    const std::vector<T> &actual) {
  if (expected.size() != actual.size()) {
    return testing::AssertionFailure()
      << "expected.size(): " << expected.size()
      << " != actual.size(): " << actual.size();
  }
  for (std::uint32_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual[i]) {
      return testing::AssertionFailure()
        << "expected[" << i << "]: " << expected[i]
        << " != actual[" << i << "]: " << actual[i];
    }
  }
  return testing::AssertionSuccess();
}

// helper to check float vector equality.
template<>
inline testing::AssertionResult vector_match(
    const std::vector<float> &expected,
    const std::vector<float> &actual) {
  if (expected.size() != actual.size()) {
    return testing::AssertionFailure()
      << "expected.size(): " << expected.size()
      << " != actual.size(): " << actual.size();
  }
  for (std::uint32_t i = 0; i < expected.size(); ++i) {
    if (!test_utils::float_eq(expected[i], actual[i])) {
      return testing::AssertionFailure()
        << "expected[" << i << "]: " << expected[i]
        << " != actual[" << i << "]: " << actual[i];
    }
  }
  return testing::AssertionSuccess();
}

// helper to check closeness of float vectors.
inline testing::AssertionResult vector_near(
    const std::vector<float> &expected,
    const std::vector<float> &actual,
    const float err) {
  if (expected.size() != actual.size()) {
    return testing::AssertionFailure()
      << "expected.size(): " << expected.size()
      << " != actual.size(): " << actual.size();
  }
  for (std::uint32_t i = 0; i < expected.size(); ++i) {
    if (!test_utils::float_near(expected[i], actual[i], err)) {
      return testing::AssertionFailure()
        << "expected[" << i << "]: " << expected[i]
        << " != actual[" << i << "]: " << actual[i];
    }
  }
  return testing::AssertionSuccess();
}

// helper to generate std::string from a byte array.
inline std::string bin_to_str(const std::initializer_list<int> data) {
  return std::string(data.begin(), data.end());
}

// helper to add all available devices.
void add_available_devices(std::vector<primitiv::Device *> &devices);
void add_available_naive_devices(std::vector<primitiv::Device *> &devices);
void add_available_cuda_devices(std::vector<primitiv::Device *> &devices);
void add_available_opencl_devices(std::vector<primitiv::Device *> &devices);

}  // namespace test_utils

#endif  // PRIMITIV_TEST_UTILS_H_
