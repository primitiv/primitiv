#ifndef PRIMITIV_TEST_UTILS_H_
#define PRIMITIV_TEST_UTILS_H_

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>
#include <gtest/gtest.h>

namespace test_utils {

// check whether or not two float values are near than the ULP-based threshold.
bool float_eq(const float a, const float b) {
  union f2i32 {
    float f;
    std::int32_t i;
  };
  static const int MAX_ULPS = 4;
  std::int32_t ai = reinterpret_cast<const f2i32 *>(&a)->i;
  if (ai < 0) ai = 0x80000000 - ai;
  std::int32_t bi = reinterpret_cast<const f2i32 *>(&b)->i;
  if (bi < 0) bi = 0x80000000 - bi;
  const std::int32_t diff = ai > bi ? ai - bi : bi - ai;
  return (diff <= MAX_ULPS);
}

// check whether or not two float values are near than the given error.
bool float_near(const float a, const float b, const float err) {
  return (a > b ? a - b : b - a) <= err;
}

// helper to check vector equality.
template<typename T>
testing::AssertionResult vector_match(
    const std::vector<T> &expected,
    const std::vector<T> &actual) {
  if (expected.size() != actual.size()) {
    return testing::AssertionFailure()
      << "expected.size(): " << expected.size()
      << " != actual.size(): " << actual.size();
  }
  for (unsigned i = 0; i < expected.size(); ++i) {
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
testing::AssertionResult vector_match(
    const std::vector<float> &expected,
    const std::vector<float> &actual) {
  if (expected.size() != actual.size()) {
    return testing::AssertionFailure()
      << "expected.size(): " << expected.size()
      << " != actual.size(): " << actual.size();
  }
  for (unsigned i = 0; i < expected.size(); ++i) {
    if (!test_utils::float_eq(expected[i], actual[i])) {
      return testing::AssertionFailure()
        << "expected[" << i << "]: " << expected[i]
        << " != actual[" << i << "]: " << actual[i];
    }
  }
  return testing::AssertionSuccess();
}

// helper to check closeness of float vectors.
testing::AssertionResult vector_near(
    const std::vector<float> &expected,
    const std::vector<float> &actual,
    const float err) {
  if (expected.size() != actual.size()) {
    return testing::AssertionFailure()
      << "expected.size(): " << expected.size()
      << " != actual.size(): " << actual.size();
  }
  for (unsigned i = 0; i < expected.size(); ++i) {
    if (!test_utils::float_near(expected[i], actual[i], err)) {
      return testing::AssertionFailure()
        << "expected[" << i << "]: " << expected[i]
        << " != actual[" << i << "]: " << actual[i];
    }
  }
  return testing::AssertionSuccess();
}

// helper to generate std::string from a byte array.
std::string bin_to_str(const std::initializer_list<int> data) {
  return std::string(data.begin(), data.end());
}

}  // namespace test_utils

#endif  // PRIMITIV_TEST_UTILS_H_
