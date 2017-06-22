#include <config.h>

#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/error.h>
#include <primitiv/initializer_impl.h>
#include <primitiv/parameter.h>
#include <test_utils.h>

using std::vector;
using test_utils::vector_match;

namespace primitiv {

class ParameterTest : public testing::Test {
protected:
  CPUDevice dev;
};

TEST_F(ParameterTest, CheckInvalid) {
  Parameter p;
  EXPECT_EQ("", p.name());
  EXPECT_EQ(Shape(), p.shape());
  EXPECT_EQ(nullptr, p.device());
  EXPECT_FALSE(p.value().valid());
  EXPECT_FALSE(p.gradient().valid());
}

TEST_F(ParameterTest, CheckNew) {
  const Shape shape {2, 2};
  Parameter p("test", shape, &dev);
  EXPECT_EQ("test", p.name());
  EXPECT_EQ(shape, p.shape());
  EXPECT_EQ(&dev, p.device());
  EXPECT_EQ(shape, p.value().shape());
  EXPECT_EQ(shape, p.gradient().shape());
}

TEST_F(ParameterTest, CheckNewWithValues) {
  const Shape shape {2, 2};
  Parameter p("test", shape, &dev, {1, 2, 3, 4});
  EXPECT_EQ("test", p.name());
  EXPECT_EQ(shape, p.shape());
  EXPECT_EQ(&dev, p.device());
  EXPECT_EQ(shape, p.value().shape());
  EXPECT_EQ(shape, p.gradient().shape());
  EXPECT_TRUE(vector_match({1, 2, 3, 4}, p.value().to_vector()));
}

TEST_F(ParameterTest, CheckNewWithInitializer) {
  const Shape shape {2, 2};
  const initializers::Constant init(42);
  Parameter p("test", shape, &dev, init);
  EXPECT_EQ("test", p.name());
  EXPECT_EQ(shape, p.shape());
  EXPECT_EQ(&dev, p.device());
  EXPECT_EQ(shape, p.value().shape());
  EXPECT_EQ(shape, p.gradient().shape());
  EXPECT_TRUE(vector_match({42, 42, 42, 42}, p.value().to_vector()));
}

TEST_F(ParameterTest, CheckMove) {
  const Shape shape {2, 2};
  Parameter p1("test", shape, &dev, {1, 2, 3, 4});

  Parameter p2 = std::move(p1);
  EXPECT_EQ("test", p2.name());
  EXPECT_EQ(shape, p2.shape());
  EXPECT_EQ(&dev, p2.device());
  EXPECT_EQ(shape, p2.value().shape());
  EXPECT_EQ(shape, p2.gradient().shape());
  EXPECT_TRUE(vector_match({1, 2, 3, 4}, p2.value().to_vector()));

  Parameter p3;
  p3 = std::move(p2);
  EXPECT_EQ("test", p3.name());
  EXPECT_EQ(shape, p3.shape());
  EXPECT_EQ(&dev, p3.device());
  EXPECT_EQ(shape, p3.value().shape());
  EXPECT_EQ(shape, p3.gradient().shape());
  EXPECT_TRUE(vector_match({1, 2, 3, 4}, p3.value().to_vector()));
}

TEST_F(ParameterTest, CheckInvalidNew) {
  EXPECT_THROW(Parameter("test", Shape({}, 3), &dev), Error);
}

TEST_F(ParameterTest, CheckResetValueByVector) {
  const Shape shape {2, 2};
  const vector<float> expected {1, 2, 3, 4};
  Parameter p("test", shape, &dev);
  p.reset_value(expected);
  EXPECT_TRUE(vector_match(expected, p.value().to_vector()));
}

TEST_F(ParameterTest, CheckResetValueByInitializer) {
  const Shape shape {2, 2};
  const initializers::Constant init(0);
  const vector<float> expected {0, 0, 0, 0};
  Parameter p("test", shape, &dev);
  p.reset_value(init);
  EXPECT_TRUE(vector_match(expected, p.value().to_vector()));
}

TEST_F(ParameterTest, CheckResetGradient) {
  const Shape shape {2, 2};
  const vector<float> expected {0, 0, 0, 0};
  Parameter p("test", shape, &dev);
  p.reset_gradient();
  EXPECT_TRUE(vector_match(expected, p.gradient().to_vector()));
}

TEST_F(ParameterTest, CheckAddValue) {
  const Shape shape {2, 2};
  const initializers::Constant init(0);
  const vector<float> diff_values1 {1, 2, 3, 4};
  const vector<float> diff_values2 {2, 4, 6, 8};
  const Tensor diff = dev.new_tensor_by_vector(shape, diff_values1);
  Parameter p("test", shape, &dev);
  p.reset_value(init);
  p.add_value(diff);
  EXPECT_TRUE(vector_match(diff_values1, p.value().to_vector()));
  p.add_value(diff);
  EXPECT_TRUE(vector_match(diff_values2, p.value().to_vector()));
}

TEST_F(ParameterTest, CheckAddGradient) {
  const Shape shape {2, 2};
  const vector<float> diff_values1 {1, 2, 3, 4};
  const vector<float> diff_values2 {2, 4, 6, 8};
  const Tensor diff = dev.new_tensor_by_vector(shape, diff_values1);
  Parameter p("test", shape, &dev);
  p.reset_gradient();
  p.add_gradient(diff);
  EXPECT_TRUE(vector_match(diff_values1, p.gradient().to_vector()));
  p.add_gradient(diff);
  EXPECT_TRUE(vector_match(diff_values2, p.gradient().to_vector()));
}

TEST_F(ParameterTest, CheckSaveLoad) {
  const Shape shape {2, 2};
  const vector<float> values {1, 2, 3, 4};
  const std::string path = "/tmp/primitiv_ParameterTest_CheckSaveLoad_p.yaml";
  const Parameter p1("test", shape, &dev, values);
  p1.save(path);
  const Parameter p2 = Parameter::load(path, &dev);
  EXPECT_EQ("test", p2.name());
  EXPECT_EQ(shape, p2.shape());
  EXPECT_TRUE(vector_match(values, p2.value().to_vector()));
}

}  // namespace primitiv
