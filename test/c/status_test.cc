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
  EXPECT_STREQ("OK", ::primitiv_Status_get_message());
  ::primitiv_Tensor *tensor;
  ASSERT_EQ(::primitiv_Status::PRIMITIV_OK, ::primitiv_Tensor_new(&tensor));
  EXPECT_STREQ("OK", ::primitiv_Status_get_message());
  ::primitiv_Device *device;
  EXPECT_EQ(::primitiv_Status::PRIMITIV_ERROR,
            ::primitiv_Tensor_device(tensor, &device));
  EXPECT_STRNE("OK", ::primitiv_Status_get_message());
}

}  // namespace internal
}  // namespace c
}  // namespace primitiv
