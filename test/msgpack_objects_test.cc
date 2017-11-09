#include <config.h>

#include <gtest/gtest.h>
#include <primitiv/msgpack/objects.h>

namespace primitiv {
namespace msgpack {
namespace objects {

class ObjectsTest : public testing::Test {};

TEST_F(ObjectsTest, CheckAlwaysFail) {
  FAIL() << "TODO: implement this test fixture.";
}

}  // namespace objects
}  // namespace msgpack
}  // namespace primitiv
