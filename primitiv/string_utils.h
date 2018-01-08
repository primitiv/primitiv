#ifndef PRIMITIV_STRING_UTILS_H_
#define PRIMITIV_STRING_UTILS_H_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
 * Some libstdc++ (e.g. Android) do not support std::to_string().
 * We support libraries that do not have std::to_string().
 */

#define DEF_TO_STRING(type, format, size) \
  inline std::string to_string(type value) { \
    char buffer[std::numeric_limits<type>::size]; \
    std::sprintf(buffer, format, value); \
    return buffer; \
  }

// int buffer's size = (digits10 + 1) + sign + '\0'
DEF_TO_STRING(int, "%d", digits10 + 3)

// unsigned buffer's size = (digits10 + 1) + '\0'
DEF_TO_STRING(unsigned, "%u", digits10 + 2)

// long buffer's size = (digits10 + 1) + sign + '\0'
DEF_TO_STRING(long, "%ld", digits10 + 3)

// unsigned long buffer's size = (digits10 + 1) + '\0'
DEF_TO_STRING(unsigned long, "%lu", digits10 + 2)

// long long buffer's size = (digits10 + 1) + sign + '\0'
DEF_TO_STRING(long long, "%lld", digits10 + 3)

// unsigned long long buffer's size
//    = (digits10 + 1) + '\0'
DEF_TO_STRING(unsigned long long, "%llu", digits10 + 2)

// float buffer's size
//    = (max_exponent10 + 1) + period + fixed precision + sign + '\0'
DEF_TO_STRING(float, "%f", max_exponent10 + 10)

// double buffer's size
//    = (max_exponent10  + 1) + period + fixed precision + sign + '\0'
DEF_TO_STRING(double, "%f", max_exponent10 + 10)

// long double buffer's size
//    = (max_exponent10  + 1) + period + fixed precision + sign + '\0'
DEF_TO_STRING(long double, "%Lf", max_exponent10 + 10)

}  // namespace string_utils
}  // namespace primitiv

#endif  // PRIMITIV_STRING_UTILS_H_
