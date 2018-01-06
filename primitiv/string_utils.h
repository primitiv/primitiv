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
 * Wrapper functions of std::to_string()
 */
#ifdef __ANDROID__
inline std::string to_string(std::uint32_t value) {
  char *c_str = nullptr;
  if (asprintf(&c_str, "%d", value) < 0) {
    THROW_ERROR("Failed to format value");
  }
  std::unique_ptr<char, decltype(&std::free)> u(c_str, std::free);
  std::string s(u.get());
  return s;
}

inline std::string to_string(float value) {
  char *c_str = nullptr;
  if (asprintf(&c_str, "%f", value) < 0) {
    THROW_ERROR("Failed to format value");
  }
  std::unique_ptr<char, decltype(&std::free)> u(c_str, std::free);
  std::string s(u.get());
  return s;
}
#else
template<typename T>
inline std::string to_string(T value) {
  return std::to_string(value);
}
#endif

}  // namespace string_utils
}  // namespace primitiv

#endif  // PRIMITIV_STRING_UTILS_H_
