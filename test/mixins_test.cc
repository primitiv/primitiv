#include <primitiv/config.h>

#include <gtest/gtest.h>

#include <primitiv/core/mixins/default_settable.h>
#include <primitiv/core/mixins/identifiable.h>
#include <primitiv/core/mixins/noncopyable.h>
#include <primitiv/core/mixins/nonmovable.h>

namespace primitiv {
namespace mixins {

class MixinsTest : public testing::Test {};

TEST_F(MixinsTest, CheckIdentifiable) {
  class TestClass : public Identifiable<TestClass> {};

  EXPECT_THROW(TestClass::get_object(0u), Error);

  {
    TestClass a;
    EXPECT_EQ(0u, a.id());
    EXPECT_EQ(&a, &TestClass::get_object(0u));
  }
  EXPECT_THROW(TestClass::get_object(0u), Error);

  {
    TestClass a;
    TestClass b;
    EXPECT_EQ(1u, a.id());
    EXPECT_EQ(2u, b.id());
    EXPECT_EQ(&a, &TestClass::get_object(1u));
    EXPECT_EQ(&b, &TestClass::get_object(2u));
  }
  EXPECT_THROW(TestClass::get_object(1u), Error);
  EXPECT_THROW(TestClass::get_object(2u), Error);
}

TEST_F(MixinsTest, CheckDefaultSettable) {
  class TestClass : public DefaultSettable<TestClass> {};

  TestClass obj0;

  EXPECT_THROW(TestClass::get_default(), Error);
  EXPECT_THROW(TestClass::get_reference_or_default(nullptr), Error);
  EXPECT_EQ(&obj0, &TestClass::get_reference_or_default(&obj0));

  {
    TestClass obj1;
    TestClass::set_default(obj1);
    EXPECT_EQ(&obj1, &TestClass::get_default());
    EXPECT_EQ(&obj1, &TestClass::get_reference_or_default(nullptr));
    EXPECT_EQ(&obj0, &TestClass::get_reference_or_default(&obj0));

    {
      TestClass obj2;
      TestClass::set_default(obj2);
      EXPECT_EQ(&obj2, &TestClass::get_default());
      EXPECT_EQ(&obj2, &TestClass::get_reference_or_default(nullptr));
      EXPECT_EQ(&obj0, &TestClass::get_reference_or_default(&obj0));

      TestClass obj3;
      TestClass::set_default(obj3);
      EXPECT_EQ(&obj3, &TestClass::get_default());
      EXPECT_EQ(&obj3, &TestClass::get_reference_or_default(nullptr));
      EXPECT_EQ(&obj0, &TestClass::get_reference_or_default(&obj0));
    }
    EXPECT_THROW(TestClass::get_default(), Error);
    EXPECT_THROW(TestClass::get_reference_or_default(nullptr), Error);
    EXPECT_EQ(&obj0, &TestClass::get_reference_or_default(&obj0));

    TestClass obj4;
    TestClass::set_default(obj4);
    EXPECT_EQ(&obj4, &TestClass::get_default());
    EXPECT_EQ(&obj4, &TestClass::get_reference_or_default(nullptr));
    EXPECT_EQ(&obj0, &TestClass::get_reference_or_default(&obj0));
  }
  EXPECT_THROW(TestClass::get_default(), Error);
  EXPECT_THROW(TestClass::get_reference_or_default(nullptr), Error);
  EXPECT_EQ(&obj0, &TestClass::get_reference_or_default(&obj0));
}

}  // namespace mixins
}  // namespace primitiv
