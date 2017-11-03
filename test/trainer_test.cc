#include <config.h>

#include <gtest/gtest.h>
#include <primitiv/error.h>
#include <primitiv/model.h>
#include <primitiv/naive_device.h>
#include <primitiv/parameter.h>
#include <primitiv/trainer_impl.h>
#include <test_utils.h>

using std::vector;
using test_utils::vector_match;

namespace primitiv {

class TrainerTest : public testing::Test {
protected:
  devices::Naive dev;
};

TEST_F(TrainerTest, CheckAddParameter) {
  Device::set_default(dev);
  trainers::SGD trainer;
  Parameter param1({2, 2});
  Parameter param2({2, 2});
  Parameter param3({2, 2});

  EXPECT_NO_THROW(trainer.add_parameter(param1));
  EXPECT_THROW(trainer.add_parameter(param1), Error);

  EXPECT_NO_THROW(trainer.add_parameter(param2));
  EXPECT_THROW(trainer.add_parameter(param1), Error);
  EXPECT_THROW(trainer.add_parameter(param2), Error);

  EXPECT_NO_THROW(trainer.add_parameter(param3));
  EXPECT_THROW(trainer.add_parameter(param1), Error);
  EXPECT_THROW(trainer.add_parameter(param2), Error);
  EXPECT_THROW(trainer.add_parameter(param3), Error);
}

TEST_F(TrainerTest, CheckAddModel) {
  Device::set_default(dev);
  trainers::SGD trainer;
  Model m;
  Parameter param1({2, 2});
  Parameter param2({2, 2});
  Parameter param3({2, 2});
  m.add_parameter("param1", param1);
  m.add_parameter("param2", param2);
  m.add_parameter("param3", param3);

  EXPECT_NO_THROW(trainer.add_model(m));
  EXPECT_THROW(trainer.add_model(m), Error);
  EXPECT_THROW(trainer.add_parameter(param1), Error);
  EXPECT_THROW(trainer.add_parameter(param2), Error);
  EXPECT_THROW(trainer.add_parameter(param3), Error);
}

TEST_F(TrainerTest, CheckAddModelWithMultipleModels) {
  Device::set_default(dev);
  trainers::SGD trainer;
  Model m1, m2, m3;
  Parameter param1({2, 2});
  Parameter param2({2, 2});
  Parameter param3({2, 2});
  m1.add_parameter("param1", param1);
  m2.add_parameter("param2", param2);
  m3.add_parameter("param3", param3);

  EXPECT_NO_THROW(trainer.add_model(m1));
  EXPECT_NO_THROW(trainer.add_model(m2));
  EXPECT_NO_THROW(trainer.add_model(m3));
  EXPECT_THROW(trainer.add_model(m1), Error);
  EXPECT_THROW(trainer.add_model(m2), Error);
  EXPECT_THROW(trainer.add_model(m3), Error);
  EXPECT_THROW(trainer.add_parameter(param1), Error);
  EXPECT_THROW(trainer.add_parameter(param2), Error);
  EXPECT_THROW(trainer.add_parameter(param3), Error);
}

TEST_F(TrainerTest, CheckAddModelWithSubmodels) {
  Device::set_default(dev);
  trainers::SGD trainer;
  Model m, sm, ssm;
  Parameter param1({2, 2});
  Parameter param2({2, 2});
  Parameter param3({2, 2});
  m.add_parameter("param1", param1);
  sm.add_parameter("param2", param2);
  ssm.add_parameter("param3", param3);
  m.add_submodel("sm", sm);
  sm.add_submodel("ssm", ssm);

  EXPECT_NO_THROW(trainer.add_model(m));
  EXPECT_THROW(trainer.add_model(m), Error);
  EXPECT_THROW(trainer.add_model(sm), Error);
  EXPECT_THROW(trainer.add_model(ssm), Error);
  EXPECT_THROW(trainer.add_parameter(param1), Error);
  EXPECT_THROW(trainer.add_parameter(param2), Error);
  EXPECT_THROW(trainer.add_parameter(param3), Error);
}

TEST_F(TrainerTest, CheckEpoch) {
  trainers::SGD trainer;
  ASSERT_EQ(0u, trainer.get_epoch());
  for (unsigned i = 1; i < 10; ++i) {
    trainer.update();
    EXPECT_EQ(i, trainer.get_epoch());
  }
  trainer.set_epoch(0);
  EXPECT_EQ(0u, trainer.get_epoch());
  trainer.set_epoch(100);
  EXPECT_EQ(100u, trainer.get_epoch());
}

TEST_F(TrainerTest, CheckLearningRateScaling) {
  trainers::SGD trainer;
  ASSERT_EQ(1.0f, trainer.get_learning_rate_scaling());

  trainer.set_learning_rate_scaling(.1);
  EXPECT_EQ(.1f, trainer.get_learning_rate_scaling());

  trainer.set_learning_rate_scaling(0);
  EXPECT_EQ(.0f, trainer.get_learning_rate_scaling());

  EXPECT_THROW(trainer.set_learning_rate_scaling(-1), Error);
}

TEST_F(TrainerTest, CheckWeightDecay) {
  Device::set_default(dev);
  trainers::SGD trainer;
  ASSERT_EQ(.0f, trainer.get_weight_decay());

  Parameter param({2, 2});
  trainer.add_parameter(param);

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
    trainer.set_weight_decay(tc.strength);
    ASSERT_EQ(tc.strength, trainer.get_weight_decay());

    param.value().reset_by_vector(tc.in_value);
    param.gradient().reset_by_vector(tc.in_grad);
    trainer.update();
    EXPECT_TRUE(vector_match(tc.out_value, param.value().to_vector()));
    EXPECT_TRUE(vector_match(tc.out_grad, param.gradient().to_vector()));
  }

  EXPECT_THROW(trainer.set_weight_decay(-1), Error);
}

TEST_F(TrainerTest, CheckGradientClipping) {
  Device::set_default(dev);
  trainers::SGD trainer;
  ASSERT_EQ(.0f, trainer.get_gradient_clipping());

  Parameter param({2, 2});
  trainer.add_parameter(param);

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
    trainer.set_gradient_clipping(tc.threshold);
    ASSERT_EQ(tc.threshold, trainer.get_gradient_clipping());

    param.value().reset_by_vector(tc.in_value);
    param.gradient().reset_by_vector(tc.in_grad);
    trainer.update();
    EXPECT_TRUE(vector_match(tc.out_value, param.value().to_vector()));
    EXPECT_TRUE(vector_match(tc.out_grad, param.gradient().to_vector()));
  }

  EXPECT_THROW(trainer.set_gradient_clipping(-1), Error);
}

}  // namespace primitiv
