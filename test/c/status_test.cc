#include <primitiv/config.h>

#include <gtest/gtest.h>
#include <primitiv/c/device.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/status.h>
#include <primitiv/c/tensor.h>

class CStatusTest : public testing::Test {};

namespace primitiv {
namespace c {
namespace internal {

TEST_F(CStatusTest, CheckMessage) {
  std::size_t length;
  ::primitiv_Status_get_message(nullptr, &length);
  char buffer1[length];
  ::primitiv_Status_get_message(buffer1, nullptr);
  EXPECT_STREQ("OK", buffer1);
  ::primitiv_Tensor *tensor;
  ASSERT_EQ(::primitiv_Status::PRIMITIV_OK, ::primitiv_Tensor_new(&tensor));
  ::primitiv_Status_get_message(nullptr, &length);
  char buffer2[length];
  ::primitiv_Status_get_message(buffer2, nullptr);
  EXPECT_STREQ("OK", buffer2);
  ::primitiv_Device *device;
  EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
            ::primitiv_Tensor_device(tensor, &device));
  ::primitiv_Status_get_message(nullptr, &length);
  char buffer3[length];
  ::primitiv_Status_get_message(buffer3, nullptr);
  EXPECT_STRNE("OK", buffer3);
}

}  // namespace internal
}  // namespace c
}  // namespace primitiv
