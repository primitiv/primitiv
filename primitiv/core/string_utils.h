#ifndef PRIMITIV_CORE_STRING_UTILS_H_
#define PRIMITIV_CORE_STRING_UTILS_H_

#include <cstdint>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace primitiv {
namespace string_utils {

/**
 * Concatenates all strings in the vector using delimiter.
 * @param strs Strings to be concatenated.
 * @param delim Delimiter.
 * @return A concatenated string with following format:
 *           0 contents        : ""
 *           1 content         : strs[0]
 *           2 contents or more: strs[0] + delim + strs[1] + delim + ...
 */
inline std::string join(
    const std::vector<std::string> &strs,
    const std::string &delim) {
  if (strs.empty()) return std::string();
  std::stringstream ss;
  ss << strs[0];
  for (std::uint32_t i = 1; i < strs.size(); ++i) ss << delim << strs[i];
  return ss.str();
}

/**
 * Imprementation of std::to_string()
 *
 * NOTE(vbkaisetsu):
 * Some libstdc++ (e.g. Android) do not support std::to_string().
 * We support libraries that do not have std::to_string().
 */

// int buffer's size = (digits10 + 1) + sign + '\0'
inline std::string to_string(int value) {
  char buffer[std::numeric_limits<int>::digits10 + 3];
  std::sprintf(buffer, "%d", value);
  return buffer;
}

// unsigned buffer's size = (digits10 + 1) + '\0'
inline std::string to_string(unsigned value) {
  char buffer[std::numeric_limits<unsigned>::digits10 + 2];
  std::sprintf(buffer, "%u", value);
  return buffer;
}

// long buffer's size = (digits10 + 1) + sign + '\0'
inline std::string to_string(long value) {
  char buffer[std::numeric_limits<long>::digits10 + 3];
  std::sprintf(buffer, "%ld", value);
  return buffer;
}

// unsigned long buffer's size = (digits10 + 1) + '\0'
inline std::string to_string(unsigned long value) {
  char buffer[std::numeric_limits<unsigned long>::digits10 + 2];
  std::sprintf(buffer, "%lu", value);
  return buffer;
}

// long long buffer's size = (digits10 + 1) + sign + '\0'
inline std::string to_string(long long value) {
  char buffer[std::numeric_limits<long long>::digits10 + 3];
  std::sprintf(buffer, "%lld", value);
  return buffer;
}

// unsigned long long buffer's size
//    = (digits10 + 1) + '\0'
inline std::string to_string(unsigned long long value) {
  char buffer[std::numeric_limits<unsigned long long>::digits10 + 2];
  std::sprintf(buffer, "%llu", value);
  return buffer;
}

// float buffer's size
//    = (max_exponent10 + 1) + period + fixed precision + sign + '\0'
inline std::string to_string(float value) {
  char buffer[std::numeric_limits<float>::max_exponent10 + 10];
  std::sprintf(buffer, "%f", value);
  return buffer;
}

// double buffer's size
//    = (max_exponent10  + 1) + period + fixed precision + sign + '\0'
inline std::string to_string(double value) {
  char buffer[std::numeric_limits<double>::max_exponent10 + 10];
  std::sprintf(buffer, "%f", value);
  return buffer;
}

// long double buffer's size
//    = (max_exponent10  + 1) + period + fixed precision + sign + '\0'
inline std::string to_string(long double value) {
  char buffer[std::numeric_limits<long double>::max_exponent10 + 10];
  std::sprintf(buffer, "%Lf", value);
  return buffer;
}

}  // namespace string_utils
}  // namespace primitiv

#endif  // PRIMITIV_CORE_STRING_UTILS_H_
