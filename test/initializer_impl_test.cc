#include <config.h>

#include <cmath>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/initializer_impl.h>
#include <primitiv/shape.h>
#include <test_utils.h>

using std::vector;

namespace primitiv {
namespace initializers {

class InitializerImplTest : public testing::Test {
protected:
  CPUDevice dev;
};

TEST_F(InitializerImplTest, CheckConstant) {
  const Shape shape {3, 3, 3};
  for (const float k : {1, 10, 100, 1000, 10000}) {
    const vector<float> expected(shape.size(), k);
    const Constant init(k);
    Tensor x = dev.new_tensor(shape);
    init.apply(x);
    EXPECT_EQ(expected, x.to_vector());
  }
}

TEST_F(InitializerImplTest, CheckXavierUniform) {
  const float scale = .0625 * std::sqrt(3);  // sqrt(6/(256+256))
  const XavierUniform init;
  Tensor x = dev.new_tensor({256, 256});
  init.apply(x);
  for (const float observed : x.to_vector()) {
    EXPECT_GE(scale, std::abs(observed));
  }
}

TEST_F(InitializerImplTest, CheckInvalidXavierUniform) {
  const XavierUniform init;
  Tensor x = dev.new_tensor({2, 2, 2});
  EXPECT_THROW(init.apply(x), std::runtime_error);
}

}  // namespace initializers
}  // namespace primitiv
