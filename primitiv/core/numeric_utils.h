#ifndef PRIMITIV_CORE_NUMERIC_UTILS_H_
#define PRIMITIV_CORE_NUMERIC_UTILS_H_

#include <cstdint>

namespace primitiv {
namespace numeric_utils {

/**
 * Calculates the minimum number of shifts `s` that satisfies `(1 << s) >= x`,
 * or formally same as `ceil(log2(x))`.
 * @param x The input number `x`.
 * @return The number of shifts `s`.
 * @remarks This function returns 64 if `x == 0`.
 */
inline std::uint64_t calculate_shifts(std::uint64_t x) {
  if (x == 0) return 64;  // Not supported

  // Flips all bits at the right of leftmost-1 to 1.
  std::uint64_t b = x | (x >> 32);
  b |= b >> 16;
  b |= b >> 8;
  b |= b >> 4;
  b |= b >> 2;
  b |= b >> 1;

  // Counts the number of 1.
  b = (b & 0x5555555555555555ull) + ((b >> 1) & 0x5555555555555555ull);
  b = (b & 0x3333333333333333ull) + ((b >> 2) & 0x3333333333333333ull);
  b = (b & 0x0f0f0f0f0f0f0f0full) + ((b >> 4) & 0x0f0f0f0f0f0f0f0full);
  b = (b & 0x00ff00ff00ff00ffull) + ((b >> 8) & 0x00ff00ff00ff00ffull);
  b = (b & 0x0000ffff0000ffffull) + ((b >> 16) & 0x0000ffff0000ffffull);
  b = (b & 0x00000000ffffffffull) + ((b >> 32) & 0x00000000ffffffffull);

  // Adjusts the result.
  return b - (1ull << (b - 1) == x);
}

}  // namespace numeric_utils
}  // namespace primitiv

#endif  // PRIMITIV_CORE_NUMERIC_UTILS_H_
