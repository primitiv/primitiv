#include <primitiv/config.h>

#include <gtest/gtest.h>

#include <primitiv/c/devices/naive/device.h>
#include <primitiv/c/parameter.h>
#include <primitiv/c/optimizer_impl.h>
#include <primitiv/c/status.h>

namespace primitiv {
namespace c {

class COptimizerTest : public testing::Test {
  void SetUp() override {
    ::primitivCreateNaiveDevice(&dev);
  }
  void TearDown() override {
    ::primitivDeleteDevice(dev);
  }
 protected:
  ::primitivDevice_t *dev;
};

TEST_F(COptimizerTest, CheckAddParameter) {
  ::primitivResetStatus();
  ::primitivSetDefaultDevice(dev);
  ::primitivOptimizer_t *optimizer;
  ::primitivCreateSgdOptimizer(0.1, &optimizer);
  ::primitivParameter_t *param1;
  ::primitivCreateParameter(&param1);
  ::primitivParameter_t *param2;
  ::primitivCreateParameter(&param2);
  ::primitivParameter_t *param3;
  ::primitivCreateParameter(&param3);

  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddParameterToOptimizer(optimizer, param1));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddParameterToOptimizer(optimizer, param1));

  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddParameterToOptimizer(optimizer, param2));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddParameterToOptimizer(optimizer, param1));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddParameterToOptimizer(optimizer, param2));

  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddParameterToOptimizer(optimizer, param3));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddParameterToOptimizer(optimizer, param1));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddParameterToOptimizer(optimizer, param2));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddParameterToOptimizer(optimizer, param3));

  ::primitivDeleteOptimizer(optimizer);
  ::primitivDeleteParameter(param1);
  ::primitivDeleteParameter(param2);
  ::primitivDeleteParameter(param3);
}

TEST_F(COptimizerTest, CheckAddModel) {
  ::primitivResetStatus();
  ::primitivSetDefaultDevice(dev);
  ::primitivOptimizer_t *optimizer;
  ::primitivCreateSgdOptimizer(0.1, &optimizer);
  ::primitivModel_t *m;
  ::primitivCreateModel(&m);
  ::primitivParameter_t *param1;
  ::primitivCreateParameter(&param1);
  ::primitivParameter_t *param2;
  ::primitivCreateParameter(&param2);
  ::primitivParameter_t *param3;
  ::primitivCreateParameter(&param3);
  ::primitivAddParameterToModel(m, "param1", param1);
  ::primitivAddParameterToModel(m, "param2", param2);
  ::primitivAddParameterToModel(m, "param3", param3);

  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddModelToOptimizer(optimizer, m));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddModelToOptimizer(optimizer, m));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddParameterToOptimizer(optimizer, param1));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddParameterToOptimizer(optimizer, param2));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddParameterToOptimizer(optimizer, param3));

  ::primitivDeleteOptimizer(optimizer);
  ::primitivDeleteModel(m);
  ::primitivDeleteParameter(param1);
  ::primitivDeleteParameter(param2);
  ::primitivDeleteParameter(param3);
}

TEST_F(COptimizerTest, CheckAddModelWithMultipleModels) {
  ::primitivResetStatus();
  ::primitivSetDefaultDevice(dev);
  ::primitivOptimizer_t *optimizer;
  ::primitivCreateSgdOptimizer(0.1, &optimizer);
  ::primitivModel_t *m1;
  ::primitivCreateModel(&m1);
  ::primitivModel_t *m2;
  ::primitivCreateModel(&m2);
  ::primitivModel_t *m3;
  ::primitivCreateModel(&m3);
  ::primitivParameter_t *param1;
  ::primitivCreateParameter(&param1);
  ::primitivParameter_t *param2;
  ::primitivCreateParameter(&param2);
  ::primitivParameter_t *param3;
  ::primitivCreateParameter(&param3);
  ::primitivAddParameterToModel(m1, "param1", param1);
  ::primitivAddParameterToModel(m2, "param2", param2);
  ::primitivAddParameterToModel(m3, "param3", param3);

  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddModelToOptimizer(optimizer, m1));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddModelToOptimizer(optimizer, m2));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddModelToOptimizer(optimizer, m3));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddModelToOptimizer(optimizer, m1));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddModelToOptimizer(optimizer, m2));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddModelToOptimizer(optimizer, m3));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddParameterToOptimizer(optimizer, param1));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddParameterToOptimizer(optimizer, param2));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitivAddParameterToOptimizer(optimizer, param3));

  ::primitivDeleteOptimizer(optimizer);
  ::primitivDeleteModel(m1);
  ::primitivDeleteModel(m2);
  ::primitivDeleteModel(m3);
  ::primitivDeleteParameter(param1);
  ::primitivDeleteParameter(param2);
  ::primitivDeleteParameter(param3);
}

TEST_F(COptimizerTest, CheckConfigs) {
  ::primitivResetStatus();
  ::primitivSetDefaultDevice(dev);
  ::primitivOptimizer_t *optimizer;
  ::primitivCreateSgdOptimizer(0.1, &optimizer);

  ::primitivExecuteOptimizerUpdate(optimizer);
  ::primitivExecuteOptimizerUpdate(optimizer);
  uint32_t uint_value = 0;
  EXPECT_EQ(PRIMITIV_C_OK, ::primitivGetOptimizerIntConfig(
      optimizer, "Optimizer.epoch", &uint_value));
  EXPECT_EQ(2u, uint_value);

  uint_value = 0;
  EXPECT_EQ(PRIMITIV_C_OK, ::primitivGetOptimizerIntConfig(
      optimizer, "undefined", &uint_value));

  EXPECT_EQ(PRIMITIV_C_OK, ::primitivSetOptimizerIntConfig(
      optimizer, "Optimizer.epoch", 10));

  uint_value = 0;
  EXPECT_EQ(PRIMITIV_C_OK, ::primitivGetOptimizerIntConfig(
      optimizer, "Optimizer.epoch", &uint_value));
  EXPECT_EQ(10u, uint_value);

  EXPECT_EQ(PRIMITIV_C_OK, ::primitivSetOptimizerIntConfig(
      optimizer, "foo", 50));

  uint_value = 1;
  EXPECT_EQ(PRIMITIV_C_OK, ::primitivGetOptimizerIntConfig(
      optimizer, "foo", &uint_value));
  EXPECT_FLOAT_EQ(1u, uint_value);

  float float_value = 0.0;
  EXPECT_EQ(PRIMITIV_C_OK, ::primitivGetOptimizerFloatConfig(
      optimizer, "SGD.eta", &float_value));
  EXPECT_FLOAT_EQ(0.1, float_value);

  float_value = 0.0;
  EXPECT_EQ(PRIMITIV_C_OK, ::primitivGetOptimizerFloatConfig(
      optimizer, "undefined", &float_value));

  EXPECT_EQ(PRIMITIV_C_OK, ::primitivSetOptimizerFloatConfig(
      optimizer, "SGD.eta", 0.2));

  float_value = 0.0;
  EXPECT_EQ(PRIMITIV_C_OK, ::primitivGetOptimizerFloatConfig(
      optimizer, "SGD.eta", &float_value));
  EXPECT_FLOAT_EQ(0.2, float_value);

  EXPECT_EQ(PRIMITIV_C_OK, ::primitivSetOptimizerFloatConfig(
      optimizer, "bar", 0.5));

  float_value = 1.0;
  EXPECT_EQ(PRIMITIV_C_OK, ::primitivGetOptimizerFloatConfig(
      optimizer, "bar", &float_value));
  EXPECT_FLOAT_EQ(1.0, float_value);

  ::primitivDeleteOptimizer(optimizer);
}

}  // namespace c
}  // namespace primitiv
