#ifndef PRIMITIV_TEST_UTILS_H_
#define PRIMITIV_TEST_UTILS_H_

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include <primitiv/core/device.h>
#include <primitiv/core/error.h>

#define IGNORE_NOT_IMPLEMENTED \
catch (const primitiv::NotImplementedError &ex) { \
  std::cerr << ex.what() << std::endl; \
} catch (const primitiv::Error &ex) { \
  if (std::strstr(ex.what(), "NOT_SUPPORTED") != 0) { \
    std::cerr << ex.what() << std::endl; \
  } \
}

namespace test_utils {

// obtains the difference of ULPs.
inline std::int32_t float_ulp_diff(float a, float b) {
  static_assert(sizeof(std::int32_t) == sizeof(float), "");
  std::int32_t ai;
  std::memcpy(&ai, &a, sizeof(std::int32_t));
  if (ai < 0) ai = 0x80000000 - ai;
  std::int32_t bi;
  std::memcpy(&bi, &b, sizeof(std::int32_t));
  if (bi < 0) bi = 0x80000000 - bi;
  return ai > bi ? ai - bi : bi - ai;
}

// check whether or not two float values are near than the ULP-based threshold.
inline bool float_eq(float a, float b, int max_ulps) {
  return test_utils::float_ulp_diff(a, b) <= max_ulps;
}

// check whether or not two float values are near than the given error.
inline bool float_near(float a, float b, float err) {
  return (a > b ? a - b : b - a) <= err;
}

// helper to check float vector equality under specified ULPs.
inline testing::AssertionResult vector_match_ulps(
    const std::vector<float> &expected,
    const std::vector<float> &actual,
    int max_ulps) {
  if (expected.size() != actual.size()) {
    return testing::AssertionFailure()
      << "expected.size(): " << expected.size()
      << " != actual.size(): " << actual.size();
  }
  for (std::uint32_t i = 0; i < expected.size(); ++i) {
    if (!test_utils::float_eq(expected[i], actual[i], max_ulps)) {
      return testing::AssertionFailure()
        << "expected[" << i << "]: " << expected[i]
        << " != actual[" << i << "]: " << actual[i]
        << " diff: " << test_utils::float_ulp_diff(expected[i], actual[i]);
    }
  }
  return testing::AssertionSuccess();
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

// float version of vector_match: using fixed ULP matching.
template<>
inline testing::AssertionResult vector_match(
    const std::vector<float> &expected,
    const std::vector<float> &actual) {
  static const int FIXED_MAX_ULPS = 4;
  return test_utils::vector_match_ulps(expected, actual, FIXED_MAX_ULPS);
}

// helper to check closeness of float vectors.
inline testing::AssertionResult vector_near(
    const std::vector<float> &expected,
    const std::vector<float> &actual,
    float err) {
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

// helper to check float array equality under specified ULPs.
inline testing::AssertionResult array_match_ulps(
    float expected[],
    float actual[],
    std::size_t n,
    int max_ulps) {
  for (std::uint32_t i = 0; i < n; ++i) {
    if (!test_utils::float_eq(expected[i], actual[i], max_ulps)) {
      return testing::AssertionFailure()
          << "expected[" << i << "]: " << expected[i]
          << " != actual[" << i << "]: " << actual[i];
    }
  }
  return testing::AssertionSuccess();
}

// helper to check array equality.
template<typename T>
inline testing::AssertionResult array_match(
    T expected[],
    T actual[],
    std::size_t n) {
  for (std::uint32_t i = 0; i < n; ++i) {
    if (expected[i] != actual[i]) {
      return testing::AssertionFailure()
          << "expected[" << i << "]: " << expected[i]
          << " != actual[" << i << "]: " << actual[i];
    }
  }
  return testing::AssertionSuccess();
}

// float version of array_match: using fixed ULP matching.
template<>
inline testing::AssertionResult array_match(
    float expected[],
    float actual[],
    std::size_t n) {
  static const int FIXED_MAX_ULPS = 4;
  return test_utils::array_match_ulps(expected, actual, n, FIXED_MAX_ULPS);
}

// helper to check closeness of float arrays.
inline testing::AssertionResult array_near(
    float expected[],
    float actual[],
    std::size_t n,
    float err) {
  for (std::uint32_t i = 0; i < n; ++i) {
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

// helper to generate vector with values {bias, bias + 1, bias + 2, ...}.
inline std::vector<float> make_iota_vector(std::size_t size, float bias) {
  std::vector<float> ret;
  ret.reserve(size);
  for (std::size_t i = 0; i < size; ++i) {
    ret.emplace_back(bias + i);
  }
  return ret;
}

// Retrieves the default boundary of ULP errors.
std::uint32_t get_default_ulps(const primitiv::Device &dev);

// helper to add all available devices.
void add_available_devices(std::vector<primitiv::Device *> &devices);
void add_available_naive_devices(std::vector<primitiv::Device *> &devices);
void add_available_eigen_devices(std::vector<primitiv::Device *> &devices);
void add_available_cuda_devices(std::vector<primitiv::Device *> &devices);
void add_available_cuda16_devices(std::vector<primitiv::Device *> &devices);
void add_available_opencl_devices(std::vector<primitiv::Device *> &devices);

}  // namespace test_utils

#endif  // PRIMITIV_TEST_UTILS_H_
