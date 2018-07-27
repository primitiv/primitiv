#include <primitiv/config.h>

#include <gtest/gtest.h>

#include <primitiv/core/error.h>
#include <primitiv/core/model.h>
#include <primitiv/devices/naive/device.h>
#include <primitiv/core/optimizer.h>
#include <primitiv/core/optimizer_impl.h>
#include <primitiv/core/parameter.h>

#include <test_utils.h>

using std::vector;
using test_utils::vector_match;

namespace primitiv {

/*
 * Optimizer class for test cases which checks numbers of function calls.
 */
class TestOptimizer : public Optimizer {
public:
  TestOptimizer(std::size_t expected) : ok_(true), expected_(expected) {}

  bool ok() const {
    return
      ok_
      && configured_.size() == expected_
      && updated_.size() == expected_;
  }

  void get_configs(
      std::unordered_map<std::string, std::uint32_t> &,
      std::unordered_map<std::string, float> &) const override {}
  void set_configs(
      const std::unordered_map<std::string, std::uint32_t> &,
      const std::unordered_map<std::string, float> &) override {}

private:
  void configure_parameter(Parameter &param) override {
    // This function should be called only once for each parameter.
    if (configured_.find(&param) != configured_.end()) ok_ = false;
    configured_.emplace(&param);
  }

  void update_parameter(float, Parameter &param) override {
    // This function should be called only once for each parameter.
    if (updated_.find(&param) != updated_.end()) ok_ = false;
    updated_.emplace(&param);
  }

  bool ok_;
  std::size_t expected_;
  std::unordered_set<Parameter *> configured_;
  std::unordered_set<Parameter *> updated_;
};

class OptimizerTest : public testing::Test {
protected:
  devices::Naive dev;
};

TEST_F(OptimizerTest, CheckAddNothing) {
  TestOptimizer optimizer(0u);
  EXPECT_NO_THROW(optimizer.add());
  EXPECT_NO_THROW(optimizer.update());
  EXPECT_TRUE(optimizer.ok());
}

TEST_F(OptimizerTest, CheckAddParameter) {
  Device::set_default(dev);
  TestOptimizer optimizer(3u);
  Parameter param1;
  Parameter param2;
  Parameter param3;

  EXPECT_NO_THROW(optimizer.add(param1));
  EXPECT_NO_THROW(optimizer.add(param1));

  EXPECT_NO_THROW(optimizer.add(param2));
  EXPECT_NO_THROW(optimizer.add(param1));
  EXPECT_NO_THROW(optimizer.add(param2));

  EXPECT_NO_THROW(optimizer.add(param3));
  EXPECT_NO_THROW(optimizer.add(param1));
  EXPECT_NO_THROW(optimizer.add(param2));
  EXPECT_NO_THROW(optimizer.add(param3));

  EXPECT_NO_THROW(optimizer.update());

  EXPECT_TRUE(optimizer.ok());
}

TEST_F(OptimizerTest, CheckAddModel) {
  Device::set_default(dev);
  TestOptimizer optimizer(3u);
  Model m;
  Parameter param1;
  Parameter param2;
  Parameter param3;
  m.add("param1", param1);
  m.add("param2", param2);
  m.add("param3", param3);

  EXPECT_NO_THROW(optimizer.add(m));
  EXPECT_NO_THROW(optimizer.add(m));
  EXPECT_NO_THROW(optimizer.add(param1));
  EXPECT_NO_THROW(optimizer.add(param2));
  EXPECT_NO_THROW(optimizer.add(param3));

  EXPECT_NO_THROW(optimizer.update());

  EXPECT_TRUE(optimizer.ok());
}

TEST_F(OptimizerTest, CheckAddModelWithMultipleModels) {
  Device::set_default(dev);
  TestOptimizer optimizer(3u);
  Model m1, m2, m3;
  Parameter param1;
  Parameter param2;
  Parameter param3;
  m1.add("param1", param1);
  m2.add("param2", param2);
  m3.add("param3", param3);

  EXPECT_NO_THROW(optimizer.add(m1));
  EXPECT_NO_THROW(optimizer.add(m2));
  EXPECT_NO_THROW(optimizer.add(m3));
  EXPECT_NO_THROW(optimizer.add(m1));
  EXPECT_NO_THROW(optimizer.add(m2));
  EXPECT_NO_THROW(optimizer.add(m3));
  EXPECT_NO_THROW(optimizer.add(param1));
  EXPECT_NO_THROW(optimizer.add(param2));
  EXPECT_NO_THROW(optimizer.add(param3));

  EXPECT_NO_THROW(optimizer.update());

  EXPECT_TRUE(optimizer.ok());
}

TEST_F(OptimizerTest, CheckAddModelWithSubmodels) {
  Device::set_default(dev);
  TestOptimizer optimizer(3u);
  Model m, sm, ssm;
  Parameter param1;
  Parameter param2;
  Parameter param3;
  m.add("param1", param1);
  sm.add("param2", param2);
  ssm.add("param3", param3);
  m.add("sm", sm);
  sm.add("ssm", ssm);

  EXPECT_NO_THROW(optimizer.add(m));
  EXPECT_NO_THROW(optimizer.add(m));
  EXPECT_NO_THROW(optimizer.add(sm));
  EXPECT_NO_THROW(optimizer.add(ssm));
  EXPECT_NO_THROW(optimizer.add(param1));
  EXPECT_NO_THROW(optimizer.add(param2));
  EXPECT_NO_THROW(optimizer.add(param3));

  EXPECT_NO_THROW(optimizer.update());

  EXPECT_TRUE(optimizer.ok());
}

TEST_F(OptimizerTest, CheckAddMultipleParameters) {
  Device::set_default(dev);
  TestOptimizer optimizer(4u);
  Parameter param1, param2, param3, param4;

  EXPECT_NO_THROW(optimizer.add(param1, param2, param3));
  EXPECT_NO_THROW(optimizer.add(param1));
  EXPECT_NO_THROW(optimizer.add(param2));
  EXPECT_NO_THROW(optimizer.add(param3));
  EXPECT_NO_THROW(optimizer.add(param1, param2));
  EXPECT_NO_THROW(optimizer.add(param2, param3));
  EXPECT_NO_THROW(optimizer.add(param1, param3));
  EXPECT_NO_THROW(optimizer.add(param1, param2, param3));
  EXPECT_NO_THROW(optimizer.add(param4));

  EXPECT_NO_THROW(optimizer.update());

  EXPECT_TRUE(optimizer.ok());
}

TEST_F(OptimizerTest, CheckAddMultipleModels) {
  Device::set_default(dev);
  TestOptimizer optimizer(4u);
  Parameter p1, p2, p3, p4;
  Model m1, m2, m3, m4;
  m1.add("p", p1);
  m2.add("p", p2);
  m3.add("p", p3);
  m4.add("p", p4);

  EXPECT_NO_THROW(optimizer.add(m1, m2, m3));
  EXPECT_NO_THROW(optimizer.add(m1));
  EXPECT_NO_THROW(optimizer.add(m2));
  EXPECT_NO_THROW(optimizer.add(m3));
  EXPECT_NO_THROW(optimizer.add(m1, m2));
  EXPECT_NO_THROW(optimizer.add(m2, m3));
  EXPECT_NO_THROW(optimizer.add(m1, m3));
  EXPECT_NO_THROW(optimizer.add(m1, m2, m3));
  EXPECT_NO_THROW(optimizer.add(m4));

  EXPECT_NO_THROW(optimizer.update());

  EXPECT_TRUE(optimizer.ok());
}

TEST_F(OptimizerTest, CheckAddParameterAndModelSimultaneously) {
  Device::set_default(dev);
  TestOptimizer optimizer(4u);
  Parameter p1, p2, p3, p4;
  Model m1, m2;
  m1.add("p", p1);
  m2.add("p", p2);

  EXPECT_NO_THROW(optimizer.add(m1, p3));
  EXPECT_NO_THROW(optimizer.add(m1));
  EXPECT_NO_THROW(optimizer.add(p3));
  EXPECT_NO_THROW(optimizer.add(m1, p3));
  EXPECT_NO_THROW(optimizer.add(m2));
  EXPECT_NO_THROW(optimizer.add(p4));

  EXPECT_NO_THROW(optimizer.update());

  EXPECT_TRUE(optimizer.ok());
}

TEST_F(OptimizerTest, CheckEpoch) {
  TestOptimizer optimizer(0u);
  ASSERT_EQ(0u, optimizer.get_epoch());
  for (std::uint32_t i = 1; i < 10; ++i) {
    optimizer.update();
    EXPECT_EQ(i, optimizer.get_epoch());
  }
  optimizer.set_epoch(0);
  EXPECT_EQ(0u, optimizer.get_epoch());
  optimizer.set_epoch(100);
  EXPECT_EQ(100u, optimizer.get_epoch());
}

TEST_F(OptimizerTest, CheckLearningRateScaling) {
  TestOptimizer optimizer(0u);
  ASSERT_EQ(1.0f, optimizer.get_learning_rate_scaling());

  optimizer.set_learning_rate_scaling(.1);
  EXPECT_EQ(.1f, optimizer.get_learning_rate_scaling());

  optimizer.set_learning_rate_scaling(0);
  EXPECT_EQ(.0f, optimizer.get_learning_rate_scaling());

  EXPECT_THROW(optimizer.set_learning_rate_scaling(-1), Error);
}

TEST_F(OptimizerTest, CheckWeightDecay) {
  Device::set_default(dev);
  optimizers::SGD optimizer;
  ASSERT_EQ(.0f, optimizer.get_weight_decay());

  Parameter param({2, 2}, {0, 0, 0, 0});
  optimizer.add(param);

  struct TestCase {
    float strength;
    vector<float> in_value;
    vector<float> in_grad;
    vector<float> out_value;
    vector<float> out_grad;
  };
  const vector<TestCase> test_cases {
    {1, {1, 2, 3, 4}, {0, 0, 0, 0}, {.9, 1.8, 2.7, 3.6}, {1, 2, 3, 4}},
    {.1, {1, 2, 3, 4}, {0, 0, 0, 0}, {.99, 1.98, 2.97, 3.96}, {.1, .2, .3, .4}},
    {0, {1, 2, 3, 4}, {0, 0, 0, 0}, {1, 2, 3, 4}, {0, 0, 0, 0}},
  };

  for (const TestCase &tc : test_cases) {
    optimizer.set_weight_decay(tc.strength);
    ASSERT_EQ(tc.strength, optimizer.get_weight_decay());

    param.value().reset_by_vector(tc.in_value);
    param.gradient().reset_by_vector(tc.in_grad);
    optimizer.update();
    EXPECT_TRUE(vector_match(tc.out_value, param.value().to_vector()));
    EXPECT_TRUE(vector_match(tc.out_grad, param.gradient().to_vector()));
  }

  EXPECT_THROW(optimizer.set_weight_decay(-1), Error);
}

TEST_F(OptimizerTest, CheckGradientClipping) {
  Device::set_default(dev);
  optimizers::SGD optimizer;
  ASSERT_EQ(.0f, optimizer.get_gradient_clipping());

  Parameter param({2, 2}, {0, 0, 0, 0});
  optimizer.add(param);

  struct TestCase {
    float threshold;
    vector<float> in_value;
    vector<float> in_grad;
    vector<float> out_value;
    vector<float> out_grad;
  };
  const vector<TestCase> test_cases {
    {4, {1, 2, 3, 4}, {1, 1, -1, -1}, {.9, 1.9, 3.1, 4.1}, {1, 1, -1, -1}},
    {4, {1, 2, 3, 4}, {2, 2, -2, -2}, {.8, 1.8, 3.2, 4.2}, {2, 2, -2, -2}},
    {4, {1, 2, 3, 4}, {3, 3, -3, -3}, {.8, 1.8, 3.2, 4.2}, {2, 2, -2, -2}},
    {2, {1, 2, 3, 4}, {1, 1, -1, -1}, {.9, 1.9, 3.1, 4.1}, {1, 1, -1, -1}},
    {2, {1, 2, 3, 4}, {2, 2, -2, -2}, {.9, 1.9, 3.1, 4.1}, {1, 1, -1, -1}},
    {2, {1, 2, 3, 4}, {3, 3, -3, -3}, {.9, 1.9, 3.1, 4.1}, {1, 1, -1, -1}},
    {0, {1, 2, 3, 4}, {1, 1, -1, -1}, {.9, 1.9, 3.1, 4.1}, {1, 1, -1, -1}},
    {0, {1, 2, 3, 4}, {2, 2, -2, -2}, {.8, 1.8, 3.2, 4.2}, {2, 2, -2, -2}},
    {0, {1, 2, 3, 4}, {3, 3, -3, -3}, {.7, 1.7, 3.3, 4.3}, {3, 3, -3, -3}},
  };

  for (const TestCase &tc : test_cases) {
    optimizer.set_gradient_clipping(tc.threshold);
    ASSERT_EQ(tc.threshold, optimizer.get_gradient_clipping());

    param.value().reset_by_vector(tc.in_value);
    param.gradient().reset_by_vector(tc.in_grad);
    optimizer.update();
    EXPECT_TRUE(vector_match(tc.out_value, param.value().to_vector()));
    EXPECT_TRUE(vector_match(tc.out_grad, param.gradient().to_vector()));
  }

  EXPECT_THROW(optimizer.set_gradient_clipping(-1), Error);
}

}  // namespace primitiv
