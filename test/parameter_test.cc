#include <config.h>

#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/parameter.h>
#include <test_utils.h>

using std::vector;

namespace primitiv {

class ParameterTest : public testing::Test {
protected:
  CPUDevice dev;
};

TEST_F(ParameterTest, CheckNew) {
  const Shape shape {2, 2};
  Parameter p(shape, &dev);
  EXPECT_EQ(shape, p.shape());
  EXPECT_EQ(&dev, p.device());
  EXPECT_EQ(shape, p.value().shape());
  EXPECT_EQ(shape, p.gradient().shape());
}

TEST_F(ParameterTest, CheckInvalidNew) {
  EXPECT_THROW(Parameter(Shape({}, 3), &dev), std::runtime_error);
}

TEST_F(ParameterTest, CheckResetValue) {
  const Shape shape {2, 2};
  Parameter p(shape, &dev);
  // not implemented
  FAIL();
}

TEST_F(ParameterTest, CheckResetGradient) {
  const Shape shape {2, 2};
  const vector<float> expected {0, 0, 0, 0};
  Parameter p(shape, &dev);
  p.reset_gradient();
  EXPECT_TRUE(test_utils::vector_match(expected, p.gradient().to_vector()));
}

TEST_F(ParameterTest, CheckAddValue) {
  const Shape shape {2, 2};
  Parameter p(shape, &dev);
  // not implemented
  FAIL();
}

TEST_F(ParameterTest, CheckAddGradient) {
  const Shape shape {2, 2};
  const vector<float> diff_values1 {1, 2, 3, 4};
  const vector<float> diff_values2 {2, 4, 6, 8};
  const Tensor diff(shape, &dev, diff_values1);
  Parameter p(shape, &dev);
  p.reset_gradient();
  p.add_gradient(diff);
  EXPECT_TRUE(test_utils::vector_match(diff_values1, p.gradient().to_vector()));
  p.add_gradient(diff);
  EXPECT_TRUE(test_utils::vector_match(diff_values2, p.gradient().to_vector()));
}

}  // namespace primitiv
