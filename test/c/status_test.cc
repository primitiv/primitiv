#include <primitiv/config.h>

#include <gtest/gtest.h>

#include <primitiv/c/device.h>
#include <primitiv/c/internal/internal.h>
#include <primitiv/c/status.h>
#include <primitiv/c/tensor.h>

class CStatusTest : public testing::Test {};

namespace primitiv {
namespace c {
namespace internal {

TEST_F(CStatusTest, CheckMessage) {
  std::size_t length;
  ::primitivGetMessage(nullptr, &length);
  char buffer1[length];
  ::primitivGetMessage(buffer1, &length);
  EXPECT_STREQ("OK", buffer1);
  ::primitivTensor_t *tensor;
  ASSERT_EQ(PRIMITIV_C_OK, ::primitivCreateTensor(&tensor));
  ::primitivGetMessage(nullptr, &length);
  char buffer2[length];
  ::primitivGetMessage(buffer2, &length);
  EXPECT_STREQ("OK", buffer2);
  ::primitivDevice_t *device;
  EXPECT_EQ(PRIMITIV_C_ERROR,
            ::primitivGetDeviceFromTensor(tensor, &device));
  ::primitivGetMessage(nullptr, &length);
  char buffer3[length];
  ::primitivGetMessage(buffer3, &length);
  EXPECT_STRNE("OK", buffer3);
}

}  // namespace internal
}  // namespace c
}  // namespace primitiv
