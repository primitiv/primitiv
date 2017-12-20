#include <primitiv/config.h>

#include <gtest/gtest.h>
#include <primitiv/error.h>
#include <primitiv/model.h>
#include <primitiv/naive_device.h>
#include <primitiv/parameter.h>
#include <primitiv/optimizer_impl.h>
#include <test_utils.h>

using std::vector;
using test_utils::vector_match;

namespace primitiv {

class OptimizerTest : public testing::Test {
protected:
  devices::Naive dev;
};

TEST_F(OptimizerTest, CheckAddNothing) {
  optimizers::SGD optimizer;
  EXPECT_NO_THROW(optimizer.add());
}

TEST_F(OptimizerTest, CheckAddParameter) {
  Device::set_default(dev);
  optimizers::SGD optimizer;
  Parameter param1;
  Parameter param2;
  Parameter param3;

  EXPECT_NO_THROW(optimizer.add(param1));
  EXPECT_THROW(optimizer.add(param1), Error);

  EXPECT_NO_THROW(optimizer.add(param2));
  EXPECT_THROW(optimizer.add(param1), Error);
  EXPECT_THROW(optimizer.add(param2), Error);

  EXPECT_NO_THROW(optimizer.add(param3));
  EXPECT_THROW(optimizer.add(param1), Error);
  EXPECT_THROW(optimizer.add(param2), Error);
  EXPECT_THROW(optimizer.add(param3), Error);
}

TEST_F(OptimizerTest, CheckAddModel) {
  Device::set_default(dev);
  optimizers::SGD optimizer;
  Model m;
  Parameter param1;
  Parameter param2;
  Parameter param3;
  m.add("param1", param1);
  m.add("param2", param2);
  m.add("param3", param3);

  EXPECT_NO_THROW(optimizer.add(m));
  EXPECT_THROW(optimizer.add(m), Error);
  EXPECT_THROW(optimizer.add(param1), Error);
  EXPECT_THROW(optimizer.add(param2), Error);
  EXPECT_THROW(optimizer.add(param3), Error);
}

TEST_F(OptimizerTest, CheckAddModelWithMultipleModels) {
  Device::set_default(dev);
  optimizers::SGD optimizer;
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
  EXPECT_THROW(optimizer.add(m1), Error);
  EXPECT_THROW(optimizer.add(m2), Error);
  EXPECT_THROW(optimizer.add(m3), Error);
  EXPECT_THROW(optimizer.add(param1), Error);
  EXPECT_THROW(optimizer.add(param2), Error);
  EXPECT_THROW(optimizer.add(param3), Error);
}

TEST_F(OptimizerTest, CheckAddModelWithSubmodels) {
  Device::set_default(dev);
  optimizers::SGD optimizer;
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
  EXPECT_THROW(optimizer.add(m), Error);
  EXPECT_THROW(optimizer.add(sm), Error);
  EXPECT_THROW(optimizer.add(ssm), Error);
  EXPECT_THROW(optimizer.add(param1), Error);
  EXPECT_THROW(optimizer.add(param2), Error);
  EXPECT_THROW(optimizer.add(param3), Error);
}

TEST_F(OptimizerTest, CheckAddMultipleParameters) {
  Device::set_default(dev);
  optimizers::SGD optimizer;
  Parameter param1, param2, param3, param4;

  EXPECT_NO_THROW(optimizer.add(param1, param2, param3));
  EXPECT_THROW(optimizer.add(param1), Error);
  EXPECT_THROW(optimizer.add(param2), Error);
  EXPECT_THROW(optimizer.add(param3), Error);
  EXPECT_THROW(optimizer.add(param1, param2), Error);
  EXPECT_THROW(optimizer.add(param2, param3), Error);
  EXPECT_THROW(optimizer.add(param1, param3), Error);
  EXPECT_THROW(optimizer.add(param1, param2, param3), Error);
  EXPECT_NO_THROW(optimizer.add(param4));
}

TEST_F(OptimizerTest, CheckAddMultipleModels) {
  Device::set_default(dev);
  optimizers::SGD optimizer;
  Parameter p1, p2, p3, p4;
  Model m1, m2, m3, m4;
  m1.add("p", p1);
  m2.add("p", p2);
  m3.add("p", p3);
  m4.add("p", p4);

  EXPECT_NO_THROW(optimizer.add(m1, m2, m3));
  EXPECT_THROW(optimizer.add(m1), Error);
  EXPECT_THROW(optimizer.add(m2), Error);
  EXPECT_THROW(optimizer.add(m3), Error);
  EXPECT_THROW(optimizer.add(m1, m2), Error);
  EXPECT_THROW(optimizer.add(m2, m3), Error);
  EXPECT_THROW(optimizer.add(m1, m3), Error);
  EXPECT_THROW(optimizer.add(m1, m2, m3), Error);
  EXPECT_NO_THROW(optimizer.add(m4));
}

TEST_F(OptimizerTest, CheckAddParameterAndModelSimultaneously) {
  Device::set_default(dev);
  optimizers::SGD optimizer;
  Parameter p1, p2, p3, p4;
  Model m1, m2;
  m1.add("p", p1);
  m2.add("p", p2);

  EXPECT_NO_THROW(optimizer.add(m1, p3));
  EXPECT_THROW(optimizer.add(m1), Error);
  EXPECT_THROW(optimizer.add(p3), Error);
  EXPECT_THROW(optimizer.add(m1, p3), Error);
  EXPECT_NO_THROW(optimizer.add(m2));
  EXPECT_NO_THROW(optimizer.add(p4));
}

TEST_F(OptimizerTest, CheckEpoch) {
  optimizers::SGD optimizer;
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
  optimizers::SGD optimizer;
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
