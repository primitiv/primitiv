#include <config.h>

#include <cmath>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/error.h>
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

TEST_F(InitializerImplTest, CheckUniform) {
  struct TestCase {
    float lower, upper, mean, variance;
  };
  const vector<TestCase> test_cases {
    {-.1, .1, 0, .04/12},
    {0, 1, .5, 1./12},
    {-1, 0, -.5, 1./12},
    {-.70710678, .70710678, 0, 2./12},
  };
  const unsigned N = 256;

  for (const auto &tc : test_cases) {
    const Uniform init(tc.lower, tc.upper);
    Tensor x = dev.new_tensor({N, N});
    init.apply(x);
    float m1 = 0, m2 = 0;
    for (const float v : x.to_vector()) {
      EXPECT_LT(tc.lower, v);
      EXPECT_GE(tc.upper, v);
      m1 += v;
      m2 += v * v;
    }
    const float mean = m1 / (N * N);
    const float variance = m2 / (N * N) - mean * mean;
    EXPECT_NEAR(tc.mean, mean, 1e-2);
    EXPECT_NEAR(tc.variance, variance, 1e-2);
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
  EXPECT_THROW(init.apply(x), Error);
}

}  // namespace initializers
}  // namespace primitiv
