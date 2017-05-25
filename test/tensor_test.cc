#include <config.h>

#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/tensor.h>
#include <test_utils.h>

using std::vector;
using test_utils::vector_match;

namespace primitiv {

class TensorTest : public testing::Test {
protected:
  CPUDevice dev;
};

TEST_F(TensorTest, CheckNewDefault) {
  const Tensor x;
  EXPECT_FALSE(x.valid());
  EXPECT_EQ(Shape(), x.shape());
  EXPECT_EQ(nullptr, x.device());
  EXPECT_EQ(nullptr, x.data());
}

TEST_F(TensorTest, CheckNew) {
  {
    const Tensor x = dev.new_tensor(Shape());
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape(), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    const vector<float> d = x.to_vector();
    EXPECT_EQ(1u, d.size());
    EXPECT_EQ(&dev, x.device());
  }
  {
    const Tensor x = dev.new_tensor(Shape {2, 3});
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape({2, 3}), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    const vector<float> d = x.to_vector();
    EXPECT_EQ(6u, d.size());
    EXPECT_EQ(&dev, x.device());
  }
  {
    const Tensor x = dev.new_tensor(Shape({2, 3}, 4));
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
    const Tensor x = dev.new_tensor({}, 1);
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape(), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    EXPECT_TRUE(vector_match(vector<float> {1}, x.to_vector()));
  }
  {
    const vector<float> data {1, 2, 3, 4, 5, 6};
    const Tensor x = dev.new_tensor({2, 3}, data);
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape({2, 3}), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    EXPECT_TRUE(vector_match(data, x.to_vector()));
  }
  {
    const vector<float> data {
      3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8,
      9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4,
    };
    const Tensor x = dev.new_tensor(Shape({2, 3}, 4), data);
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape({2, 3}, 4), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    EXPECT_TRUE(vector_match(data, x.to_vector()));
  }
}

TEST_F(TensorTest, CheckMove) {
  // c-tor
  {
    Tensor tmp = dev.new_tensor(Shape({2}, 3), {1, 2, 3, 4, 5, 6});
    void *ptr = tmp.data();

    Tensor x = std::move(tmp);

    EXPECT_TRUE(x.valid());
    EXPECT_FALSE(tmp.valid());
    EXPECT_EQ(Shape({2}, 3), x.shape());
    EXPECT_EQ(ptr, x.data());
    EXPECT_TRUE(vector_match({1, 2, 3, 4, 5, 6}, x.to_vector()));
  }

  // operator= (move to valid tensor)
  {
    Tensor tmp = dev.new_tensor({6}, {2, 4, 6, 8, 10 ,12});
    void *ptr = tmp.data();

    Tensor x = dev.new_tensor({}, 1);
    x = std::move(tmp);

    EXPECT_TRUE(x.valid());
    EXPECT_FALSE(tmp.valid());
    EXPECT_EQ(Shape({6}), x.shape());
    EXPECT_EQ(ptr, x.data());
    EXPECT_TRUE(vector_match({2, 4, 6, 8, 10 ,12}, x.to_vector()));
  }

  // operator= (move to invalid tensor)
  {
    Tensor tmp = dev.new_tensor(Shape({1}, 6), {3, 6, 9, 12, 15, 18});
    void *ptr = tmp.data();

    Tensor x;
    x = std::move(tmp);

    EXPECT_TRUE(x.valid());
    EXPECT_FALSE(tmp.valid());
    EXPECT_EQ(Shape({1}, 6), x.shape());
    EXPECT_EQ(ptr, x.data());
    EXPECT_TRUE(vector_match({3, 6, 9, 12, 15, 18}, x.to_vector()));
  }
}

TEST_F(TensorTest, CheckAddGradient) {
  {
    const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const vector<float> b_data {0, -1, -2, -3, -3, -4, -5, -6, -6, -7, -8, -9};
    const vector<float> y_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    Tensor a = dev.new_tensor(Shape({2, 2}, 3), a_data);
    const Tensor b = dev.new_tensor(Shape({2, 2}, 3), b_data);
    a.add_gradient(b);
    EXPECT_TRUE(vector_match(y_data, a.to_vector()));
  }
  {
    const vector<float> a_data {1, 2, 3, 4};
    const vector<float> b_data {0, -1, -2, -3, -3, -4, -5, -6, -6, -7, -8, -9};
    const vector<float> y_data {-8, -10, -12, -14};
    Tensor a = dev.new_tensor({2, 2}, a_data);
    const Tensor b = dev.new_tensor(Shape({2, 2}, 3), b_data);
    a.add_gradient(b);
    EXPECT_TRUE(vector_match(y_data, a.to_vector()));
  }
  {
    const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const vector<float> b_data {0, -1, -2, -3};
    const vector<float> y_data {1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9};
    Tensor a = dev.new_tensor(Shape({2, 2}, 3), a_data);
    const Tensor b = dev.new_tensor({2, 2}, b_data);
    a.add_gradient(b);
    EXPECT_TRUE(vector_match(y_data, a.to_vector()));
  }
}

TEST_F(TensorTest, CheckInvalidAddGradient) {
  vector<Shape> shapes {
    Shape(),
    Shape({}, 3),
    Shape({2, 2}, 2),
  };
  Tensor a = dev.new_tensor(Shape({2, 2}, 3));

  for (const Shape &shape : shapes) {
    Tensor b = dev.new_tensor(shape);
    EXPECT_THROW(a.add_gradient(b), std::runtime_error);
  }
}

}  // namespace primitiv
