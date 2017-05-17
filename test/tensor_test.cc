#include <config.h>

#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/tensor.h>
#include <test_utils.h>

using std::vector;

namespace primitiv {

class TensorTest : public testing::Test {
protected:
  CPUDevice dev;
};

TEST_F(TensorTest, CheckNewDefault) {
  const Tensor x;
  EXPECT_FALSE(x.valid());
}

TEST_F(TensorTest, CheckNew) {
  {
    const Tensor x(Shape({}), &dev);
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape({}), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    const vector<float> d = x.to_vector();
    EXPECT_EQ(1u, d.size());
    EXPECT_EQ(&dev, x.device());
  }
  {
    const Tensor x({2, 3}, &dev);
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape({2, 3}), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    const vector<float> d = x.to_vector();
    EXPECT_EQ(6u, d.size());
    EXPECT_EQ(&dev, x.device());
  }
  {
    const Tensor x(Shape({2, 3}, 4), &dev);
    EXPECT_TRUE(x.valid());
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
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape({}), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    EXPECT_TRUE(test_utils::vector_match(data, x.to_vector()));
  }
  {
    const vector<float> data {1, 2, 3, 4, 5, 6};
    const Tensor x({2, 3}, &dev, data);
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape({2, 3}), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    EXPECT_TRUE(test_utils::vector_match(data, x.to_vector()));
  }
  {
    const vector<float> data {
      3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8,
      9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4,
    };
    const Tensor x(Shape({2, 3}, 4), &dev, data);
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape({2, 3}, 4), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    EXPECT_TRUE(test_utils::vector_match(data, x.to_vector()));
  }
}

TEST_F(TensorTest, CheckMove) {
  // c-tor
  {
    Tensor tmp(Shape({2}, 3), &dev, {1, 2, 3, 4, 5, 6});
    void *ptr = tmp.data();

    Tensor x = std::move(tmp);

    EXPECT_TRUE(x.valid());
    EXPECT_FALSE(tmp.valid());
    EXPECT_EQ(Shape({2}, 3), x.shape());
    EXPECT_EQ(ptr, x.data());
    EXPECT_TRUE(test_utils::vector_match({1, 2, 3, 4, 5, 6}, x.to_vector()));
  }

  // operator= (move to valid tensor)
  {
    Tensor tmp(Shape({6}), &dev, {2, 4, 6, 8, 10 ,12});
    void *ptr = tmp.data();

    Tensor x(Shape(), &dev, {1});
    x = std::move(tmp);

    EXPECT_TRUE(x.valid());
    EXPECT_FALSE(tmp.valid());
    EXPECT_EQ(Shape({6}), x.shape());
    EXPECT_EQ(ptr, x.data());
    EXPECT_TRUE(test_utils::vector_match({2, 4, 6, 8, 10 ,12}, x.to_vector()));
  }

  // operator= (move to invalid tensor)
  {
    Tensor tmp(Shape({1}, 6), &dev, {3, 6, 9, 12, 15, 18});
    void *ptr = tmp.data();

    Tensor x;
    x = std::move(tmp);

    EXPECT_TRUE(x.valid());
    EXPECT_FALSE(tmp.valid());
    EXPECT_EQ(Shape({1}, 6), x.shape());
    EXPECT_EQ(ptr, x.data());
    EXPECT_TRUE(test_utils::vector_match({3, 6, 9, 12, 15, 18}, x.to_vector()));
  }
}

TEST_F(TensorTest, CheckAugment) {
  const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> b_data {0, -1, -2, -3, -3, -4, -5, -6, -6, -7, -8, -9};
  const vector<float> y_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  Tensor a(Shape({2, 2}, 3), &dev, a_data);
  const Tensor b(Shape({2, 2}, 3), &dev, b_data);
  a += b;
  EXPECT_TRUE(test_utils::vector_match(y_data, a.to_vector()));
}

TEST_F(TensorTest, CheckInvalidAugment) {
  vector<Shape> shapes {
    Shape(),
    Shape({}, 3),
    Shape({2, 2}),
    Shape({2, 2}, 3),
  };
  Tensor a(Shape({2, 2, 3}), &dev);

  for (const Shape &shape : shapes) {
    Tensor b(shape, &dev);
    EXPECT_THROW(a += b, std::runtime_error);
  }
}

}  // namespace primitiv
