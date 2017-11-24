#include <config.h>

#include <thread>
#include <gtest/gtest.h>
#include <primitiv/mixins.h>

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

TEST_F(MixinsTest, CheckDefaultSettableMultithread) {
  class TestClass : public DefaultSettable<TestClass> {};

  ASSERT_THROW(TestClass::get_default(), Error);
  TestClass obj_th0;
  TestClass::set_default(obj_th0);
  ASSERT_EQ(&obj_th0, &TestClass::get_default());

  std::thread th1([&] {
      EXPECT_THROW(TestClass::get_default(), Error);
      TestClass obj_th1;
      TestClass::set_default(obj_th1);
      EXPECT_EQ(&obj_th1, &TestClass::get_default());
  });
  std::thread th2([&] {
      EXPECT_THROW(TestClass::get_default(), Error);
      TestClass obj_th2;
      TestClass::set_default(obj_th2);
      EXPECT_EQ(&obj_th2, &TestClass::get_default());
  });

  EXPECT_EQ(&obj_th0, &TestClass::get_default());

  th1.join();
  th2.join();

  EXPECT_EQ(&obj_th0, &TestClass::get_default());
}

}  // namespace mixins
}  // namespace primitiv
