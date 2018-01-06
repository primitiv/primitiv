#include <primitiv/config.h>

#include <gtest/gtest.h>
#include <primitiv/c/naive_device.h>
#include <primitiv/c/parameter.h>
#include <primitiv/c/optimizer_impl.h>
#include <primitiv/c/status.h>

namespace primitiv {
namespace c {

class COptimizerTest : public testing::Test {
  void SetUp() override {
    ::primitiv_devices_Naive_new(&dev);
  }
  void TearDown() override {
    ::primitiv_Device_delete(dev);
  }
 protected:
  ::primitiv_Device *dev;
};

TEST_F(COptimizerTest, CheckAddParameter) {
  ::primitiv_reset();
  ::primitiv_Device_set_default(dev);
  ::primitiv_Optimizer *optimizer;
  ::primitiv_optimizers_SGD_new(0.1, &optimizer);
  ::primitiv_Parameter *param1;
  ::primitiv_Parameter_new(&param1);
  ::primitiv_Parameter *param2;
  ::primitiv_Parameter_new(&param2);
  ::primitiv_Parameter *param3;
  ::primitiv_Parameter_new(&param3);

  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitiv_Optimizer_add_parameter(optimizer, param1));
  EXPECT_EQ(PRIMITIV_C_ERROR,
      ::primitiv_Optimizer_add_parameter(optimizer, param1));

  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitiv_Optimizer_add_parameter(optimizer, param2));
  EXPECT_EQ(PRIMITIV_C_ERROR,
      ::primitiv_Optimizer_add_parameter(optimizer, param1));
  EXPECT_EQ(PRIMITIV_C_ERROR,
      ::primitiv_Optimizer_add_parameter(optimizer, param2));

  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitiv_Optimizer_add_parameter(optimizer, param3));
  EXPECT_EQ(PRIMITIV_C_ERROR,
      ::primitiv_Optimizer_add_parameter(optimizer, param1));
  EXPECT_EQ(PRIMITIV_C_ERROR,
      ::primitiv_Optimizer_add_parameter(optimizer, param2));
  EXPECT_EQ(PRIMITIV_C_ERROR,
      ::primitiv_Optimizer_add_parameter(optimizer, param3));

  ::primitiv_Optimizer_delete(optimizer);
  ::primitiv_Parameter_delete(param1);
  ::primitiv_Parameter_delete(param2);
  ::primitiv_Parameter_delete(param3);
}

TEST_F(COptimizerTest, CheckAddModel) {
  ::primitiv_reset();
  ::primitiv_Device_set_default(dev);
  ::primitiv_Optimizer *optimizer;
  ::primitiv_optimizers_SGD_new(0.1, &optimizer);
  ::primitiv_Model *m;
  ::primitiv_Model_new(&m);
  ::primitiv_Parameter *param1;
  ::primitiv_Parameter_new(&param1);
  ::primitiv_Parameter *param2;
  ::primitiv_Parameter_new(&param2);
  ::primitiv_Parameter *param3;
  ::primitiv_Parameter_new(&param3);
  ::primitiv_Model_add_parameter(m, "param1", param1);
  ::primitiv_Model_add_parameter(m, "param2", param2);
  ::primitiv_Model_add_parameter(m, "param3", param3);

  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitiv_Optimizer_add_model(optimizer, m));
  EXPECT_EQ(PRIMITIV_C_ERROR,
      ::primitiv_Optimizer_add_model(optimizer, m));
  EXPECT_EQ(PRIMITIV_C_ERROR,
      ::primitiv_Optimizer_add_parameter(optimizer, param1));
  EXPECT_EQ(PRIMITIV_C_ERROR,
      ::primitiv_Optimizer_add_parameter(optimizer, param2));
  EXPECT_EQ(PRIMITIV_C_ERROR,
      ::primitiv_Optimizer_add_parameter(optimizer, param3));

  ::primitiv_Optimizer_delete(optimizer);
  ::primitiv_Model_delete(m);
  ::primitiv_Parameter_delete(param1);
  ::primitiv_Parameter_delete(param2);
  ::primitiv_Parameter_delete(param3);
}

TEST_F(COptimizerTest, CheckAddModelWithMultipleModels) {
  ::primitiv_reset();
  ::primitiv_Device_set_default(dev);
  ::primitiv_Optimizer *optimizer;
  ::primitiv_optimizers_SGD_new(0.1, &optimizer);
  ::primitiv_Model *m1;
  ::primitiv_Model_new(&m1);
  ::primitiv_Model *m2;
  ::primitiv_Model_new(&m2);
  ::primitiv_Model *m3;
  ::primitiv_Model_new(&m3);
  ::primitiv_Parameter *param1;
  ::primitiv_Parameter_new(&param1);
  ::primitiv_Parameter *param2;
  ::primitiv_Parameter_new(&param2);
  ::primitiv_Parameter *param3;
  ::primitiv_Parameter_new(&param3);
  ::primitiv_Model_add_parameter(m1, "param1", param1);
  ::primitiv_Model_add_parameter(m2, "param2", param2);
  ::primitiv_Model_add_parameter(m3, "param3", param3);

  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitiv_Optimizer_add_model(optimizer, m1));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitiv_Optimizer_add_model(optimizer, m2));
  EXPECT_EQ(PRIMITIV_C_OK,
      ::primitiv_Optimizer_add_model(optimizer, m3));
  EXPECT_EQ(PRIMITIV_C_ERROR,
      ::primitiv_Optimizer_add_model(optimizer, m1));
  EXPECT_EQ(PRIMITIV_C_ERROR,
      ::primitiv_Optimizer_add_model(optimizer, m2));
  EXPECT_EQ(PRIMITIV_C_ERROR,
      ::primitiv_Optimizer_add_model(optimizer, m3));
  EXPECT_EQ(PRIMITIV_C_ERROR,
      ::primitiv_Optimizer_add_parameter(optimizer, param1));
  EXPECT_EQ(PRIMITIV_C_ERROR,
      ::primitiv_Optimizer_add_parameter(optimizer, param2));
  EXPECT_EQ(PRIMITIV_C_ERROR,
      ::primitiv_Optimizer_add_parameter(optimizer, param3));

  ::primitiv_Optimizer_delete(optimizer);
  ::primitiv_Model_delete(m1);
  ::primitiv_Model_delete(m2);
  ::primitiv_Model_delete(m3);
  ::primitiv_Parameter_delete(param1);
  ::primitiv_Parameter_delete(param2);
  ::primitiv_Parameter_delete(param3);
}

TEST_F(COptimizerTest, CheckConfigs) {
  ::primitiv_reset();
  ::primitiv_Device_set_default(dev);
  ::primitiv_Optimizer *optimizer;
  ::primitiv_optimizers_SGD_new(0.1, &optimizer);

  ::primitiv_Optimizer_update(optimizer);
  ::primitiv_Optimizer_update(optimizer);
  uint32_t uint_value = 0;
  EXPECT_EQ(PRIMITIV_C_OK, ::primitiv_Optimizer_get_int_config(
      optimizer, "Optimizer.epoch", &uint_value));
  EXPECT_EQ(2, uint_value);

  uint_value = 0;
  EXPECT_EQ(PRIMITIV_C_OK, ::primitiv_Optimizer_get_int_config(
      optimizer, "undefined", &uint_value));

  EXPECT_EQ(PRIMITIV_C_OK, ::primitiv_Optimizer_set_int_config(
      optimizer, "Optimizer.epoch", 10));

  uint_value = 0;
  EXPECT_EQ(PRIMITIV_C_OK, ::primitiv_Optimizer_get_int_config(
      optimizer, "Optimizer.epoch", &uint_value));
  EXPECT_EQ(10, uint_value);

  EXPECT_EQ(PRIMITIV_C_OK, ::primitiv_Optimizer_set_int_config(
      optimizer, "foo", 50));

  uint_value = 1;
  EXPECT_EQ(PRIMITIV_C_OK, ::primitiv_Optimizer_get_int_config(
      optimizer, "foo", &uint_value));
  EXPECT_FLOAT_EQ(1, uint_value);

  float float_value = 0.0;
  EXPECT_EQ(PRIMITIV_C_OK, ::primitiv_Optimizer_get_float_config(
      optimizer, "SGD.eta", &float_value));
  EXPECT_FLOAT_EQ(0.1, float_value);

  float_value = 0.0;
  EXPECT_EQ(PRIMITIV_C_OK, ::primitiv_Optimizer_get_float_config(
      optimizer, "undefined", &float_value));

  EXPECT_EQ(PRIMITIV_C_OK, ::primitiv_Optimizer_set_float_config(
      optimizer, "SGD.eta", 0.2));

  float_value = 0.0;
  EXPECT_EQ(PRIMITIV_C_OK, ::primitiv_Optimizer_get_float_config(
      optimizer, "SGD.eta", &float_value));
  EXPECT_FLOAT_EQ(0.2, float_value);

  EXPECT_EQ(PRIMITIV_C_OK, ::primitiv_Optimizer_set_float_config(
      optimizer, "bar", 0.5));

  float_value = 1.0;
  EXPECT_EQ(PRIMITIV_C_OK, ::primitiv_Optimizer_get_float_config(
      optimizer, "bar", &float_value));
  EXPECT_FLOAT_EQ(1.0, float_value);

  ::primitiv_Optimizer_delete(optimizer);
}

}  // namespace c
}  // namespace primitiv
