#include <primitiv/config.h>

#include <cstdint>

#include <vector>

#include <gtest/gtest.h>

#include <primitiv/core/numeric_utils.h>

namespace primitiv {
namespace numeric_utils {

class NumericUtilsTest : public testing::Test {};

TEST_F(NumericUtilsTest, CheckCalculateShifts) {
  std::vector<std::uint64_t> samples {
    64,  // 0
    0, 1, 2, 2, 3, 3, 3, 3, 4, 4,  // 1 -- 10
    4, 4, 4, 4, 4, 4, 5, 5, 5, 5,  // 11 -- 20
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,  // 21 -- 30
    5, 5, 6, 6, 6, 6, 6, 6, 6, 6,  // 31 -- 40
  };
  for (std::uint64_t i = 0; i < samples.size(); ++i) {
    EXPECT_EQ(samples[i], calculate_shifts(i));
  }
  EXPECT_EQ(62ull, calculate_shifts(0x4000000000000000ull));
  EXPECT_EQ(63ull, calculate_shifts(0x4000000000000001ull));
  EXPECT_EQ(63ull, calculate_shifts(0x8000000000000000ull));
  EXPECT_EQ(64ull, calculate_shifts(0x8000000000000001ull));
  EXPECT_EQ(64ull, calculate_shifts(0xffffffffffffffffull));
}

}  // namespace numeric_utils
}  // namespace primitiv
