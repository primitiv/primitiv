#include <config.h>

#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/tensor.h>

using std::vector;

namespace {

// helper to check vector equality.
template <typename T>
::testing::AssertionResult vector_match(
    const vector<T> &expected,
    const vector<T> &actual) {
  if (expected.size() != actual.size()) {
    return ::testing::AssertionFailure()
      << "expected.size() (" << expected.size()
      << ") != actual.size() (" << actual.size() << ")";
  }
  for (unsigned i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual[i]) {
      return ::testing::AssertionFailure()
        << "expected[" << i << "] (" << expected[i]
        << ") != actual[" << i << "] (" << actual[i] << ")";
    }
  }
  return ::testing::AssertionSuccess();
}

}  // namespace

namespace primitiv {

class TensorTest : public testing::Test {
protected:
  CPUDevice dev;
};

TEST_F(TensorTest, CheckNew) {
  {
    const Tensor x(Shape({}), &dev);
    EXPECT_EQ(Shape({}), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    const vector<float> d = x.to_vector();
    EXPECT_EQ(1u, d.size());
    EXPECT_EQ(&dev, x.device());
  }
  {
    const Tensor x({2, 3}, &dev);
    EXPECT_EQ(Shape({2, 3}), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    const vector<float> d = x.to_vector();
    EXPECT_EQ(6u, d.size());
    EXPECT_EQ(&dev, x.device());
  }
  {
    const Tensor x(Shape({2, 3}, 4), &dev);
    EXPECT_EQ(Shape({2, 3}, 4), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    const vector<float> d = x.to_vector();
    EXPECT_EQ(24u, d.size());
    EXPECT_EQ(&dev, x.device());
  }
}

TEST_F(TensorTest, CheckNewWithData) {
  {
    const vector<float> data {1};
    const Tensor x(Shape({}), &dev, data);
    EXPECT_EQ(Shape({}), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    EXPECT_TRUE(::vector_match(data, x.to_vector()));
  }
  {
    const vector<float> data {1, 2, 3, 4, 5, 6};
    const Tensor x({2, 3}, &dev, data);
    EXPECT_EQ(Shape({2, 3}), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    EXPECT_TRUE(::vector_match(data, x.to_vector()));
  }
  {
    const vector<float> data {
      3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8,
      9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4,
    };
    const Tensor x(Shape({2, 3}, 4), &dev, data);
    EXPECT_EQ(Shape({2, 3}, 4), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    EXPECT_TRUE(::vector_match(data, x.to_vector()));
  }
}

TEST_F(TensorTest, CheckMove) {
  Tensor tmp1(Shape({2}, 3), &dev, {1, 2, 3, 4, 5, 6});
  void *ptr1 = tmp1.data();
  Tensor tmp2(Shape({6}), &dev, {2, 4, 6, 8, 10 ,12});
  void *ptr2 = tmp2.data();

  Tensor x(std::move(tmp1));
  EXPECT_EQ(Shape({2}, 3), x.shape());
  EXPECT_EQ(ptr1, x.data());
  EXPECT_TRUE(::vector_match({1, 2, 3, 4, 5, 6}, x.to_vector()));

  x = std::move(tmp2);
  EXPECT_EQ(Shape({6}), x.shape());
  EXPECT_EQ(ptr2, x.data());
  EXPECT_TRUE(::vector_match({2, 4, 6, 8, 10 ,12}, x.to_vector()));
}

}  // namespace primitiv
