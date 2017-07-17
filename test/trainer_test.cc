#include <config.h>

#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/error.h>
#include <primitiv/parameter.h>
#include <primitiv/trainer_impl.h>
#include <test_utils.h>

using std::vector;
using test_utils::vector_match;

namespace primitiv {

class TrainerTest : public testing::Test {
protected:
  CPUDevice dev;
};

TEST_F(TrainerTest, CheckAddParameter) {
  trainers::SGD trainer;
  Parameter param1("param1", {2, 2}, dev);
  Parameter param2("param2", {2, 2}, dev);
  Parameter param3("param3", {2, 2}, dev);

  EXPECT_NO_THROW(trainer.add_parameter(param1));
  EXPECT_THROW(trainer.add_parameter(param1), Error);

  EXPECT_NO_THROW(trainer.add_parameter(param2));
  EXPECT_THROW(trainer.add_parameter(param1), Error);
  EXPECT_THROW(trainer.add_parameter(param2), Error);

  EXPECT_NO_THROW(trainer.add_parameter(param3));
  EXPECT_THROW(trainer.add_parameter(param1), Error);
  EXPECT_THROW(trainer.add_parameter(param2), Error);
  EXPECT_THROW(trainer.add_parameter(param3), Error);

  // Different object but same name
  Parameter param4("param1", {}, dev);
  EXPECT_THROW(trainer.add_parameter(param4), Error);
}

TEST_F(TrainerTest, CheckWeightDecay) {
  trainers::SGD trainer;
  trainer.set_weight_decay(.1);

  Parameter param("param1", {2, 2}, {1, 2, 3, 4}, dev);
  trainer.add_parameter(param);

  trainer.reset_gradients();
  trainer.update(1);
  EXPECT_TRUE(vector_match(
        vector<float> {.99, 1.98, 2.97, 3.96}, param.value().to_vector()));
  EXPECT_TRUE(vector_match(
        vector<float> {.1, .2, .3, .4}, param.gradient().to_vector()));
}

TEST_F(TrainerTest, CheckGradientClipping) {
  trainers::SGD trainer;
  trainer.set_gradient_clipping(4);

  Parameter param("param1", {2, 2}, dev);
  trainer.add_parameter(param);

  // Norm: 2 = sqrt(4*1^2) ... no clipping
  param.value().reset_by_vector({1, 2, 3, 4});
  param.gradient().reset_by_vector({1, 1, -1, -1});
  trainer.update(1);
  EXPECT_TRUE(vector_match(
        vector<float> {.9, 1.9, 3.1, 4.1}, param.value().to_vector()));
  EXPECT_TRUE(vector_match(
        vector<float> {1, 1, -1, -1}, param.gradient().to_vector()));

  // Norm: 4 = sqrt(4*2^2) ... threshold
  param.value().reset_by_vector({1, 2, 3, 4});
  param.gradient().reset_by_vector({2, 2, -2, -2});
  trainer.update(1);
  EXPECT_TRUE(vector_match(
        vector<float> {.8, 1.8, 3.2, 4.2}, param.value().to_vector()));
  EXPECT_TRUE(vector_match(
        vector<float> {2, 2, -2, -2}, param.gradient().to_vector()));

  // Norm: 6 = sqrt(4*3^2) ... clipping
  param.value().reset_by_vector({1, 2, 3, 4});
  param.gradient().reset_by_vector({3, 3, -3, -3});
  trainer.update(1);
  EXPECT_TRUE(vector_match(
        vector<float> {.8, 1.8, 3.2, 4.2}, param.value().to_vector()));
  EXPECT_TRUE(vector_match(
        vector<float> {2, 2, -2, -2}, param.gradient().to_vector()));
}

}  // namespace primitiv
