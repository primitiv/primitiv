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

class TraingerTest : public testing::Test {
protected:
  CPUDevice dev;
};

TEST_F(TraingerTest, CheckAddParameter) {
  trainers::SGD trainer(.1);
  Parameter param1("param1", {2, 2}, &dev);
  Parameter param2("param2", {2, 2}, &dev);
  Parameter param3("param3", {2, 2}, &dev);

  EXPECT_NO_THROW(trainer.add_parameter(&param1));
  EXPECT_THROW(trainer.add_parameter(&param1), Error);
  
  EXPECT_NO_THROW(trainer.add_parameter(&param2));
  EXPECT_THROW(trainer.add_parameter(&param1), Error);
  EXPECT_THROW(trainer.add_parameter(&param2), Error);
  
  EXPECT_NO_THROW(trainer.add_parameter(&param3));
  EXPECT_THROW(trainer.add_parameter(&param1), Error);
  EXPECT_THROW(trainer.add_parameter(&param2), Error);
  EXPECT_THROW(trainer.add_parameter(&param3), Error);

  // Different object but same name
  Parameter param4("param1", {}, &dev);
  EXPECT_THROW(trainer.add_parameter(&param4), Error);
}

}  // namespace primitiv
