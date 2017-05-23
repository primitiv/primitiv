#include <config.h>

#include <stdexcept>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/parameter.h>
#include <primitiv/sgd_trainer.h>
#include <test_utils.h>

using std::vector;
using test_utils::vector_match;

namespace primitiv {

class SGDTrainerTest : public testing::Test {
protected:
  CPUDevice dev;
};

TEST_F(SGDTrainerTest, CheckAddParameter) {
  SGDTrainer trainer(.1);
  Parameter param({2, 2}, &dev);
  EXPECT_NO_THROW(trainer.add_parameter(&param));
  EXPECT_THROW(trainer.add_parameter(&param), std::runtime_error);
}

TEST_F(SGDTrainerTest, CheckUpdate) {
  SGDTrainer trainer(.1);
  Parameter param({2, 2}, &dev);
  param.reset_value({1, 2, 3, 4});
  trainer.add_parameter(&param);
  trainer.reset_gradients();

  EXPECT_TRUE(vector_match(
        vector<float> {1, 2, 3, 4}, param.value().get_values()));
  EXPECT_TRUE(vector_match(
        vector<float>(4, 0), param.gradient().get_values()));

  param.add_gradient(dev.new_tensor({2, 2}, {1, 1, 1, 1}));
  trainer.update();

  EXPECT_TRUE(vector_match(
        vector<float> {.9, 1.9, 2.9, 3.9}, param.value().get_values()));
  EXPECT_TRUE(vector_match(
        vector<float>(4, 1), param.gradient().get_values()));

  param.reset_gradient();

  EXPECT_TRUE(vector_match(
        vector<float> {.9, 1.9, 2.9, 3.9}, param.value().get_values()));
  EXPECT_TRUE(vector_match(
        vector<float>(4, 0), param.gradient().get_values()));
}

}  // namespace primitiv
