#ifndef PRIMITIV_STRING_UTILS_H_
#define PRIMITIV_STRING_UTILS_H_

#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <primitiv/error.h>

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
inline std::string to_string(std::uint32_t value) {
  /*
   * buffer's size = (digits10 + 1) + '\0'
   *               =    9      + 1  +  1
   */
  char buffer[std::numeric_limits<std::uint32_t>::digits10 + 2];
  std::sprintf(buffer, "%u", value);
  return buffer;
}

inline std::string to_string(float value) {
  /*
   * buffer's size
   *   = (max_exponent10 + 1) + period + fixed precision + sign + '\0'
   *   =      38         + 1  +   1    +        6        +  1   +  1
   */
  char buffer[std::numeric_limits<float>::max_exponent10 + 10];
  std::sprintf(buffer, "%f", value);
  return buffer;
}

}  // namespace string_utils
}  // namespace primitiv

#endif  // PRIMITIV_STRING_UTILS_H_
