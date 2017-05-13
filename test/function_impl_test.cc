#include <config.h>

#include <gtest/gtest.h>
#include <primitiv/function_impl.h>

namespace primitiv {
namespace functions {

class FunctionImplTest : public testing::Test {};

TEST_F(FunctionImplTest, CheckFail) {
  FAIL();
}

}  // namespace functions
}  // namespace primitiv
