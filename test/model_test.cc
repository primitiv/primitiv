#include <config.h>

#include <gtest/gtest.h>
#include <primitiv/model.h>
#include <primitiv/parameter.h>

namespace primitiv {

class ModelTest : public testing::Test {};

TEST_F(ModelTest, CheckAddParameter) {
  Model m;
  Parameter p1, p2, p3;

  EXPECT_NO_THROW(m.add_parameter("p1", p1));
  EXPECT_THROW(m.add_parameter("p1", p1), Error);
  EXPECT_THROW(m.add_parameter("x", p1), Error);

  EXPECT_THROW(m.add_parameter("p1", p2), Error);
  EXPECT_NO_THROW(m.add_parameter("p2", p2));
  EXPECT_THROW(m.add_parameter("p2", p2), Error);
  EXPECT_THROW(m.add_parameter("x", p2), Error);

  EXPECT_THROW(m.add_parameter("p1", p3), Error);
  EXPECT_THROW(m.add_parameter("p2", p3), Error);
  EXPECT_NO_THROW(m.add_parameter("p3", p3));
  EXPECT_THROW(m.add_parameter("p3", p3), Error);
  EXPECT_THROW(m.add_parameter("x", p3), Error);
}

}  // namespace primitiv
