#include <config.h>

#include <gtest/gtest.h>
#include <primitiv/default_scope.h>
#include <primitiv/error.h>
#include <primitiv/graph.h>
#include <primitiv/naive_device.h>

namespace primitiv {

class DefaultScopeTest : public testing::Test {};

TEST_F(DefaultScopeTest, CheckDifferentClasses) {
  devices::Naive dev;
  Graph g;

  EXPECT_EQ(0u, DefaultScope<Device>::size());
  EXPECT_EQ(0u, DefaultScope<Graph>::size());
  {
    DefaultScope<Device> ds(dev);
    EXPECT_EQ(1u, DefaultScope<Device>::size());
    EXPECT_EQ(0u, DefaultScope<Graph>::size());
  }
  {
    DefaultScope<Graph> gs(g);
    EXPECT_EQ(0u, DefaultScope<Device>::size());
    EXPECT_EQ(1u, DefaultScope<Graph>::size());
  }
  EXPECT_EQ(0u, DefaultScope<Device>::size());
  EXPECT_EQ(0u, DefaultScope<Graph>::size());
}

TEST_F(DefaultScopeTest, CheckHierarchy) {
  devices::Naive dev1, dev2;

  EXPECT_EQ(0u, DefaultScope<Device>::size());
  EXPECT_THROW(DefaultScope<Device>::get(), Error);
  {
    DefaultScope<Device> ds(dev1);
    EXPECT_EQ(1u, DefaultScope<Device>::size());
    EXPECT_EQ(&dev1, &DefaultScope<Device>::get());
    {
      DefaultScope<Device> ds(dev2);
      EXPECT_EQ(2u, DefaultScope<Device>::size());
      EXPECT_EQ(&dev2, &DefaultScope<Device>::get());
      {
        DefaultScope<Device> ds(dev1);
        EXPECT_EQ(3u, DefaultScope<Device>::size());
        EXPECT_EQ(&dev1, &DefaultScope<Device>::get());
        DefaultScope<Device> ds2;  // invalid scope
        EXPECT_EQ(4u, DefaultScope<Device>::size());
        EXPECT_THROW(&DefaultScope<Device>::get(), Error);
      }
      EXPECT_EQ(2u, DefaultScope<Device>::size());
      EXPECT_EQ(&dev2, &DefaultScope<Device>::get());
    }
    EXPECT_EQ(1u, DefaultScope<Device>::size());
    EXPECT_EQ(&dev1, &DefaultScope<Device>::get());
  }
  EXPECT_EQ(0u, DefaultScope<Device>::size());
  EXPECT_THROW(DefaultScope<Device>::get(), Error);
}

}  // namespace primitiv
