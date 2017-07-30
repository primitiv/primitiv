#include <config.h>

#include <cstdio>
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
  EXPECT_FALSE(p.valid());
  EXPECT_THROW(p.name(), Error);
  EXPECT_THROW(p.has_stats(""), Error);
  EXPECT_THROW(p.shape(), Error);
  EXPECT_THROW(p.device(), Error);
  EXPECT_THROW(p.value(), Error);
  EXPECT_THROW(p.gradient(), Error);
}

TEST_F(ParameterTest, CheckNew) {
  Device::set_default_device(dev);
  const Shape shape {2, 2};
  Parameter p("test", shape);
  EXPECT_TRUE(p.valid());
  EXPECT_EQ("test", p.name());
  EXPECT_EQ(shape, p.shape());
  EXPECT_EQ(&dev, &p.device());
  EXPECT_EQ(shape, p.value().shape());
  EXPECT_EQ(shape, p.gradient().shape());
}

TEST_F(ParameterTest, CheckNewWithValues) {
  Device::set_default_device(dev);
  const Shape shape {2, 2};
  Parameter p("test", shape, {1, 2, 3, 4});
  EXPECT_TRUE(p.valid());
  EXPECT_EQ("test", p.name());
  EXPECT_EQ(shape, p.shape());
  EXPECT_EQ(&dev, &p.device());
  EXPECT_EQ(shape, p.value().shape());
  EXPECT_EQ(shape, p.gradient().shape());
  EXPECT_TRUE(vector_match({1, 2, 3, 4}, p.value().to_vector()));
}

TEST_F(ParameterTest, CheckNewWithInitializer) {
  Device::set_default_device(dev);
  const Shape shape {2, 2};
  const initializers::Constant init(42);
  Parameter p("test", shape, init);
  EXPECT_TRUE(p.valid());
  EXPECT_EQ("test", p.name());
  EXPECT_EQ(shape, p.shape());
  EXPECT_EQ(&dev, &p.device());
  EXPECT_EQ(shape, p.value().shape());
  EXPECT_EQ(shape, p.gradient().shape());
  EXPECT_TRUE(vector_match({42, 42, 42, 42}, p.value().to_vector()));
}

TEST_F(ParameterTest, CheckAddStats) {
  Device::set_default_device(dev);
  Parameter p("test", {});
  EXPECT_FALSE(p.has_stats("a"));
  EXPECT_FALSE(p.has_stats("b"));
  EXPECT_THROW(p.stats("a"), std::out_of_range);
  EXPECT_THROW(p.stats("b"), std::out_of_range);

  p.add_stats("a", {2, 2});
  EXPECT_TRUE(p.has_stats("a"));
  EXPECT_FALSE(p.has_stats("b"));
  EXPECT_THROW(p.stats("b"), std::out_of_range);

  Tensor &a = p.stats("a");
  EXPECT_EQ(Shape({2, 2}), a.shape());
  a.reset_by_vector({1, 2, 3, 4});

  p.add_stats("b", {3, 3});
  EXPECT_TRUE(p.has_stats("a"));
  EXPECT_TRUE(p.has_stats("b"));

  Tensor &a2 = p.stats("a");
  Tensor &b2 = p.stats("b");
  EXPECT_EQ(&a, &a2);
  EXPECT_EQ(Shape({2, 2}), a2.shape());
  EXPECT_EQ(Shape({3, 3}), b2.shape());
  EXPECT_TRUE(vector_match({1, 2, 3, 4}, a2.to_vector()));

  EXPECT_THROW(p.add_stats("a", {}), Error);
  EXPECT_THROW(p.add_stats("b", {}), Error);
}

TEST_F(ParameterTest, CheckInvalidAddStats) {
  Parameter p;
  EXPECT_THROW(p.add_stats("a", {}), Error);
}

TEST_F(ParameterTest, CheckMove) {
  Device::set_default_device(dev);
  const Shape shape {2, 2};
  Parameter p1("test", shape, {1, 2, 3, 4});
  ASSERT_TRUE(p1.valid());

  Parameter p2 = std::move(p1);
  ASSERT_FALSE(p1.valid());
  ASSERT_TRUE(p2.valid());
  EXPECT_EQ("test", p2.name());
  EXPECT_EQ(shape, p2.shape());
  EXPECT_EQ(&dev, &p2.device());
  EXPECT_EQ(shape, p2.value().shape());
  EXPECT_EQ(shape, p2.gradient().shape());
  EXPECT_TRUE(vector_match({1, 2, 3, 4}, p2.value().to_vector()));

  Parameter p3;
  p3 = std::move(p2);
  ASSERT_FALSE(p2.valid());
  ASSERT_TRUE(p3.valid());
  EXPECT_EQ("test", p3.name());
  EXPECT_EQ(shape, p3.shape());
  EXPECT_EQ(&dev, &p3.device());
  EXPECT_EQ(shape, p3.value().shape());
  EXPECT_EQ(shape, p3.gradient().shape());
  EXPECT_TRUE(vector_match({1, 2, 3, 4}, p3.value().to_vector()));
}

TEST_F(ParameterTest, CheckInvalidNew) {
  Device::set_default_device(dev);
  EXPECT_THROW(Parameter("test", Shape({}, 3)), Error);
}

TEST_F(ParameterTest, CheckResetValueByVector) {
  Device::set_default_device(dev);
  const Shape shape {2, 2};
  const vector<float> expected {1, 2, 3, 4};
  Parameter p("test", shape);
  p.reset_value(expected);
  EXPECT_TRUE(vector_match(expected, p.value().to_vector()));
}

TEST_F(ParameterTest, CheckResetValueByInitializer) {
  Device::set_default_device(dev);
  const Shape shape {2, 2};
  const initializers::Constant init(0);
  const vector<float> expected {0, 0, 0, 0};
  Parameter p("test", shape);
  p.reset_value(init);
  EXPECT_TRUE(vector_match(expected, p.value().to_vector()));
}

TEST_F(ParameterTest, CheckResetGradient) {
  Device::set_default_device(dev);
  const Shape shape {2, 2};
  const vector<float> expected {0, 0, 0, 0};
  Parameter p("test", shape);
  p.reset_gradient();
  EXPECT_TRUE(vector_match(expected, p.gradient().to_vector()));
}

TEST_F(ParameterTest, CheckAddValue) {
  Device::set_default_device(dev);
  const Shape shape {2, 2};
  const initializers::Constant init(0);
  const vector<float> diff_values1 {1, 2, 3, 4};
  const vector<float> diff_values2 {2, 4, 6, 8};
  const Tensor diff = dev.new_tensor_by_vector(shape, diff_values1);
  Parameter p("test", shape);
  p.reset_value(init);
  p.value() += diff;
  EXPECT_TRUE(vector_match(diff_values1, p.value().to_vector()));
  p.value() += diff;
  EXPECT_TRUE(vector_match(diff_values2, p.value().to_vector()));
}

TEST_F(ParameterTest, CheckAddGradient) {
  Device::set_default_device(dev);
  const Shape shape {2, 2};
  const vector<float> diff_values1 {1, 2, 3, 4};
  const vector<float> diff_values2 {2, 4, 6, 8};
  const Tensor diff = dev.new_tensor_by_vector(shape, diff_values1);
  Parameter p("test", shape);
  p.reset_gradient();
  p.gradient() += diff;
  EXPECT_TRUE(vector_match(diff_values1, p.gradient().to_vector()));
  p.gradient() += diff;
  EXPECT_TRUE(vector_match(diff_values2, p.gradient().to_vector()));
}

TEST_F(ParameterTest, CheckSaveLoad) {
  Device::set_default_device(dev);
  const Shape shape {2, 2};
  const vector<float> values {1, 2, 3, 4};
  const Parameter p1("test", shape, values);

  const std::string path = "/tmp/primitiv_ParameterTest_CheckSaveLoad.data";
  p1.save(path);

  const Parameter p2 = Parameter::load(path);
  std::remove(path.c_str());

  EXPECT_EQ("test", p2.name());
  EXPECT_EQ(shape, p2.shape());
  EXPECT_TRUE(vector_match(values, p2.value().to_vector()));
}

TEST_F(ParameterTest, CheckSaveLoadWithStats) {
  Device::set_default_device(dev);
  const Shape shape {2, 2};
  const vector<float> values {1, 2, 3, 4};
  Parameter p1("test", shape, values);
  p1.add_stats("a", {2, 2});
  p1.stats("a").reset_by_vector(values);

  const std::string path = "/tmp/primitiv_ParameterTest_CheckSaveLoadWithStats.data";
  p1.save(path);

  const Parameter p2 = Parameter::load(path);
  std::remove(path.c_str());

  EXPECT_EQ("test", p2.name());
  EXPECT_EQ(shape, p2.shape());
  EXPECT_TRUE(vector_match(values, p2.value().to_vector()));
  ASSERT_TRUE(p2.has_stats("a"));
  EXPECT_TRUE(vector_match(values, p2.stats("a").to_vector()));
}

TEST_F(ParameterTest, CheckSaveWithoutStats) {
  Device::set_default_device(dev);
  const Shape shape {2, 2};
  const vector<float> values {1, 2, 3, 4};
  Parameter p1("test", shape, values);
  p1.add_stats("a", {2, 2});
  p1.stats("a").reset_by_vector(values);

  const std::string path = "/tmp/primitiv_ParameterTest_CheckSaveWithoutStats.data";
  p1.save(path, false);

  const Parameter p2 = Parameter::load(path);
  std::remove(path.c_str());

  EXPECT_EQ("test", p2.name());
  EXPECT_EQ(shape, p2.shape());
  EXPECT_TRUE(vector_match(values, p2.value().to_vector()));
  ASSERT_FALSE(p2.has_stats("a"));
}

TEST_F(ParameterTest, CheckLoadWithoutStats) {
  Device::set_default_device(dev);
  const Shape shape {2, 2};
  const vector<float> values {1, 2, 3, 4};
  Parameter p1("test", shape, values);
  p1.add_stats("a", {2, 2});
  p1.stats("a").reset_by_vector(values);

  const std::string path = "/tmp/primitiv_ParameterTest_CheckLoadWithoutStats.data";
  p1.save(path);

  const Parameter p2 = Parameter::load(path, false);
  std::remove(path.c_str());

  EXPECT_EQ("test", p2.name());
  EXPECT_EQ(shape, p2.shape());
  EXPECT_TRUE(vector_match(values, p2.value().to_vector()));
  ASSERT_FALSE(p2.has_stats("a"));
}

TEST_F(ParameterTest, CheckInvalidSave) {
  Parameter invalid;
  EXPECT_THROW(invalid.save("/tmp/not_generated"), Error);
}

}  // namespace primitiv
