#include <primitiv/config.h>

#include <gtest/gtest.h>
#include <primitiv/compiler.h>
#include <primitiv/naive_device.h>
#include <test_utils.h>

namespace primitiv {

class CUDACompilerTest : public testing::Test {};

TEST_F(CUDACompilerTest, Temporary) {
  Compiler cp(3, 1);
  auto a = cp.input(0);
  auto b = cp.input(1);
  auto c = cp.input(2);
  cp.output(0, a + b + c);

  devices::Naive dev;
  cp.compile(dev, "/tmp");
}

}  // namespace primitiv
