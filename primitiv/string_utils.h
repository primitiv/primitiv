#ifndef PRIMITIV_STRING_UTILS_H_
#define PRIMITIV_STRING_UTILS_H_

#include <cstdint>
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
std::string join(
    const std::vector<std::string> &strs,
    const std::string &delim) {
  if (strs.empty()) return std::string();
  std::stringstream ss;
  ss << strs[0];
  for (std::uint32_t i = 1; i < strs.size(); ++i) ss << delim << strs[i];
  return ss.str();
}

}  // namespace string_utils
}  // namespace primitiv

#endif  // PRIMITIV_STRING_UTILS_H_
