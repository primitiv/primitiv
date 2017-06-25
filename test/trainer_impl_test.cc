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

class TraingerImplTest : public testing::Test {
protected:
  CPUDevice dev;
};

TEST_F(TraingerImplTest, CheckSGDUpdate) {
  SGDTrainer trainer(.1);
  Parameter param("param", {2, 2}, {1, 2, 3, 4}, &dev);
  trainer.add_parameter(&param);
  trainer.reset_gradients();

  EXPECT_TRUE(vector_match(
        vector<float> {1, 2, 3, 4}, param.value().to_vector()));
  EXPECT_TRUE(vector_match(
        vector<float>(4, 0), param.gradient().to_vector()));

  param.add_gradient(dev.new_tensor_by_vector({2, 2}, {1, 1, 1, 1}));
  trainer.update(1);

  EXPECT_TRUE(vector_match(
        vector<float> {.9, 1.9, 2.9, 3.9}, param.value().to_vector()));
  EXPECT_TRUE(vector_match(
        vector<float>(4, 1), param.gradient().to_vector()));

  param.reset_gradient();

  EXPECT_TRUE(vector_match(
        vector<float> {.9, 1.9, 2.9, 3.9}, param.value().to_vector()));
  EXPECT_TRUE(vector_match(
        vector<float>(4, 0), param.gradient().to_vector()));
}

}  // namespace primitiv
