#include <config.h>

#include <gtest/gtest.h>
#include <primitiv/mixins.h>

namespace primitiv {
namespace mixins {

class MixinsTest : public testing::Test {};

TEST_F(MixinsTest, CheckDefaultSettable) {
  class TestClass : public DefaultSettable<TestClass> {};

  EXPECT_THROW(TestClass::get_default(), Error);
  {
    TestClass obj1;
    TestClass::set_default(obj1);
    EXPECT_EQ(&obj1, &TestClass::get_default());

    {
      TestClass obj2;
      TestClass::set_default(obj2);
      EXPECT_EQ(&obj2, &TestClass::get_default());

      TestClass obj3;
      TestClass::set_default(obj3);
      EXPECT_EQ(&obj3, &TestClass::get_default());
    }
    EXPECT_THROW(TestClass::get_default(), Error);

    TestClass obj4;
    TestClass::set_default(obj4);
    EXPECT_EQ(&obj4, &TestClass::get_default());
  }
  EXPECT_THROW(TestClass::get_default(), Error);
}

}  // namespace mixins
}  // namespace primitiv
