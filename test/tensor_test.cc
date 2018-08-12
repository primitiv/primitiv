#include <primitiv/config.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <primitiv/core/arithmetic.h>
#include <primitiv/core/error.h>
#include <primitiv/devices/naive/device.h>
#include <primitiv/core/tensor.h>

#include <test_utils.h>

using std::vector;
using test_utils::vector_match;

namespace primitiv {

class TensorTest : public testing::Test {
protected:
  static vector<Device *> devices;

  static void SetUpTestCase() {
    test_utils::add_available_devices(devices);
  }

  static void TearDownTestCase() {
    for (Device *dev : devices) {
      delete dev;
    }
  }
};

vector<Device *> TensorTest::devices;

TEST_F(TensorTest, CheckInvalid) {
  const Tensor x;
  EXPECT_FALSE(x.valid());
  EXPECT_THROW(x.shape(), Error);
  EXPECT_THROW(x.device(), Error);
  EXPECT_THROW(x.to_float(), Error);
  EXPECT_THROW(x.to_vector(), Error);
}

TEST_F(TensorTest, CheckInvalidate) {
  for (Device *dev : devices) {
    Tensor x = dev->new_tensor_by_constant({}, 1);
    ASSERT_TRUE(x.valid());
    x.invalidate();
    EXPECT_FALSE(x.valid());
  }
}

TEST_F(TensorTest, CheckNewScalarWithData) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor_by_constant({}, 1);
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(dev, &x.device());
    EXPECT_EQ(Shape(), x.shape());
    EXPECT_TRUE(vector_match(vector<float> {1}, x.to_vector()));
    EXPECT_FLOAT_EQ(1.0f, x.to_float());
  }
}

TEST_F(TensorTest, CheckNewMatrixWithData) {
  for (Device *dev : devices) {
    const vector<float> data {1, 2, 3, 4, 5, 6};
    const Tensor x = dev->new_tensor_by_vector({2, 3}, data);
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(dev, &x.device());
    EXPECT_EQ(Shape({2, 3}), x.shape());
    EXPECT_TRUE(vector_match(data, x.to_vector()));
    EXPECT_THROW(x.to_float(), Error);
  }
}

TEST_F(TensorTest, CheckNewMatrixMinibatchWithData) {
  for (Device *dev : devices) {
    const vector<float> data {
      3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8,
      9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4,
    };
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 4), data);
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(dev, &x.device());
    EXPECT_EQ(Shape({2, 3}, 4), x.shape());
    EXPECT_TRUE(vector_match(data, x.to_vector()));
    EXPECT_THROW(x.to_float(), Error);
  }
}

TEST_F(TensorTest, CheckMoveValidToNew) {
  for (Device *dev : devices) {
    Tensor tmp = dev->new_tensor_by_vector(Shape({2}, 3), {1, 2, 3, 4, 5, 6});

    const Tensor x = std::move(tmp);
    EXPECT_TRUE(x.valid());
    EXPECT_FALSE(tmp.valid());
    EXPECT_EQ(Shape({2}, 3), x.shape());
    EXPECT_TRUE(vector_match({1, 2, 3, 4, 5, 6}, x.to_vector()));
  }
}

TEST_F(TensorTest, CheckMoveValidToValid) {
  for (Device *dev : devices) {
    Tensor tmp = dev->new_tensor_by_vector({6}, {2, 4, 6, 8, 10 ,12});

    Tensor x = dev->new_tensor_by_constant({}, 1);
    ASSERT_TRUE(x.valid());

    x = std::move(tmp);
    EXPECT_TRUE(x.valid());
    EXPECT_FALSE(tmp.valid());
    EXPECT_EQ(Shape({6}), x.shape());
    EXPECT_TRUE(vector_match({2, 4, 6, 8, 10 ,12}, x.to_vector()));
  }
}

TEST_F(TensorTest, CheckMoveValidToInvalid) {
  for (Device *dev : devices) {
    Tensor tmp = dev->new_tensor_by_vector(
        Shape({1}, 6), {3, 6, 9, 12, 15, 18});

    Tensor x;
    ASSERT_FALSE(x.valid());

    x = std::move(tmp);
    EXPECT_TRUE(x.valid());
    EXPECT_FALSE(tmp.valid());
    EXPECT_EQ(Shape({1}, 6), x.shape());
    EXPECT_TRUE(vector_match({3, 6, 9, 12, 15, 18}, x.to_vector()));
  }
}

#if 0
// Some compilers does not compile this test due to "-Wself-move".
TEST_F(TensorTest, CheckMoveValidToThis) {
  for (Device *dev : devices) {
    Tensor x = dev->new_tensor_by_vector({6}, {2, 4, 6, 8, 10 ,12});

    x = std::move(x);
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape({6}), x.shape());
    EXPECT_TRUE(vector_match({2, 4, 6, 8, 10 ,12}, x.to_vector()));
  }
}
#endif

TEST_F(TensorTest, CheckMoveInvalidToNew) {
  Tensor tmp;
  ASSERT_FALSE(tmp.valid());

  const Tensor x = std::move(tmp);
  EXPECT_FALSE(x.valid());
  EXPECT_FALSE(tmp.valid());
}

TEST_F(TensorTest, CheckMoveInvalidToValid) {
  for (Device *dev : devices) {
    Tensor tmp;
    ASSERT_FALSE(tmp.valid());

    Tensor x = dev->new_tensor_by_constant({}, 1);
    ASSERT_TRUE(x.valid());

    x = std::move(tmp);
    EXPECT_FALSE(x.valid());
    EXPECT_FALSE(tmp.valid());
  }
}

TEST_F(TensorTest, CheckMoveInvalidToInalid) {
  Tensor tmp;
  ASSERT_FALSE(tmp.valid());

  Tensor x;
  ASSERT_FALSE(x.valid());

  x = std::move(tmp);
  EXPECT_FALSE(x.valid());
  EXPECT_FALSE(tmp.valid());
}

#if 0
// Some compilers does not compile this test due to "-Wself-move".
TEST_F(TensorTest, CheckMoveInvalidToThis) {
  Tensor x;
  ASSERT_FALSE(x.valid());

  x = std::move(x);
  EXPECT_FALSE(x.valid());
}
#endif

TEST_F(TensorTest, CheckCopyValidToNew) {
  for (Device *dev : devices) {
    const Tensor tmp = dev->new_tensor_by_vector(Shape({2}, 3), {1, 2, 3, 4, 5, 6});

    const Tensor x = tmp;
    EXPECT_TRUE(x.valid());
    EXPECT_TRUE(tmp.valid());
    EXPECT_EQ(Shape({2}, 3), x.shape());
    EXPECT_EQ(Shape({2}, 3), tmp.shape());
    EXPECT_TRUE(vector_match({1, 2, 3, 4, 5, 6}, x.to_vector()));
    EXPECT_TRUE(vector_match({1, 2, 3, 4, 5, 6}, tmp.to_vector()));
  }
}

TEST_F(TensorTest, CheckCopyValidToValid) {
  for (Device *dev : devices) {
    const Tensor tmp = dev->new_tensor_by_vector({6}, {2, 4, 6, 8, 10 ,12});

    Tensor x = dev->new_tensor_by_constant({}, 1);
    ASSERT_TRUE(x.valid());

    x = tmp;
    EXPECT_TRUE(x.valid());
    EXPECT_TRUE(tmp.valid());
    EXPECT_EQ(Shape({6}), x.shape());
    EXPECT_EQ(Shape({6}), tmp.shape());
    EXPECT_TRUE(vector_match({2, 4, 6, 8, 10 ,12}, x.to_vector()));
    EXPECT_TRUE(vector_match({2, 4, 6, 8, 10 ,12}, tmp.to_vector()));
  }
}

TEST_F(TensorTest, CheckCopyValidToInvalid) {
  for (Device *dev : devices) {
    const Tensor tmp = dev->new_tensor_by_vector(
        Shape({1}, 6), {3, 6, 9, 12, 15, 18});

    Tensor x;
    ASSERT_FALSE(x.valid());

    x = tmp;
    EXPECT_TRUE(x.valid());
    EXPECT_TRUE(tmp.valid());
    EXPECT_EQ(Shape({1}, 6), x.shape());
    EXPECT_EQ(Shape({1}, 6), tmp.shape());
    EXPECT_TRUE(vector_match({3, 6, 9, 12, 15, 18}, x.to_vector()));
    EXPECT_TRUE(vector_match({3, 6, 9, 12, 15, 18}, tmp.to_vector()));
  }
}

TEST_F(TensorTest, CheckCopyValidToThis) {
  for (Device *dev : devices) {
    Tensor x = dev->new_tensor_by_vector({6}, {2, 4, 6, 8, 10 ,12});

    x = x;
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape({6}), x.shape());
    EXPECT_TRUE(vector_match({2, 4, 6, 8, 10 ,12}, x.to_vector()));
  }
}

TEST_F(TensorTest, CheckCopyInvalidToNew) {
  const Tensor tmp;
  ASSERT_FALSE(tmp.valid());

  const Tensor x = tmp;
  EXPECT_FALSE(x.valid());
  EXPECT_FALSE(tmp.valid());
}

TEST_F(TensorTest, CheckCopyInvalidToValid) {
  for (Device *dev : devices) {
    const Tensor tmp;
    ASSERT_FALSE(tmp.valid());

    Tensor x = dev->new_tensor_by_constant({}, 1);
    ASSERT_TRUE(x.valid());

    x = tmp;
    EXPECT_FALSE(x.valid());
    EXPECT_FALSE(tmp.valid());
  }
}

TEST_F(TensorTest, CheckCopyInvalidToInalid) {
  const Tensor tmp;
  ASSERT_FALSE(tmp.valid());

  Tensor x;
  ASSERT_FALSE(x.valid());

  x = tmp;
  EXPECT_FALSE(x.valid());
  EXPECT_FALSE(tmp.valid());
}

TEST_F(TensorTest, CheckCopyInvalidToThis) {
  Tensor x;
  ASSERT_FALSE(x.valid());

  x = x;
  EXPECT_FALSE(x.valid());
}

TEST_F(TensorTest, CheckResetValuesByConstant) {
  for (Device *dev : devices) {
    {
      const Tensor x = dev->new_tensor_by_constant(Shape({2, 2}, 2), 42);
      EXPECT_TRUE(vector_match(vector<float>(8, 42), x.to_vector()));
    }
    {
      Tensor x = dev->new_tensor_by_constant(Shape({2, 2}, 2), 0);

      x.reset(42);
      EXPECT_TRUE(vector_match(vector<float>(8, 42), x.to_vector()));
    }
    {
      Tensor x = dev->new_tensor_by_constant(Shape({2, 2}, 2), 123);
      const Tensor copied = x;

      x.reset(42);
      EXPECT_TRUE(vector_match(vector<float>(8, 42), x.to_vector()));
      EXPECT_TRUE(vector_match(vector<float>(8, 123), copied.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckResetValuesByArray) {
  for (Device *dev : devices) {
    {
      const float data[] {1, 2, 3, 4, 5, 6, 7, 8};
      const Tensor x = dev->new_tensor_by_array(Shape({2, 2}, 2), data);
      EXPECT_TRUE(
          vector_match(vector<float> {1, 2, 3, 4, 5, 6, 7, 8}, x.to_vector()));
    }
    {
      const float data[] {1, 2, 3, 4, 5, 6, 7, 8};
      Tensor x = dev->new_tensor_by_constant(Shape({2, 2}, 2), 0);
      x.reset_by_array(data);
      EXPECT_TRUE(
          vector_match(vector<float> {1, 2, 3, 4, 5, 6, 7, 8}, x.to_vector()));
    }
    {
      const float data[] {1, 2, 3, 4, 5, 6, 7, 8};
      Tensor x = dev->new_tensor_by_constant(Shape({2, 2}, 2), 123);
      const Tensor copied = x;

      x.reset_by_array(data);
      EXPECT_TRUE(
          vector_match(vector<float> {1, 2, 3, 4, 5, 6, 7, 8}, x.to_vector()));
      EXPECT_TRUE(vector_match(vector<float>(8, 123), copied.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckResetValuesByVector) {
  for (Device *dev : devices) {
    {
      const vector<float> data {1, 2, 3, 4, 5, 6, 7, 8};
      const Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 2), data);
      EXPECT_TRUE(vector_match(data, x.to_vector()));
    }
    {
      const vector<float> data {1, 2, 3, 4, 5, 6, 7, 8};
      Tensor x = dev->new_tensor_by_constant(Shape({2, 2}, 2), 0);
      x.reset_by_vector(data);
      EXPECT_TRUE(vector_match(data, x.to_vector()));
    }
    {
      const vector<float> data {1, 2, 3, 4, 5, 6, 7, 8};
      Tensor x = dev->new_tensor_by_constant(Shape({2, 2}, 2), 123);
      const Tensor copied = x;

      x.reset_by_vector(data);
      EXPECT_TRUE(vector_match(data, x.to_vector()));
      EXPECT_TRUE(vector_match(vector<float>(8, 123), copied.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckInplaceMultiplyConst) {
  const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> y1_data {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24};
  const vector<float> y2_data {.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6};

  for (Device *dev : devices) {
    {
      Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 3), x_data);
      x.inplace_multiply_const(2);
      EXPECT_TRUE(vector_match(y1_data, x.to_vector()));
    }
    {
      Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 3), x_data);
      x *= 2;
      EXPECT_TRUE(vector_match(y1_data, x.to_vector()));
    }
    {
      Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 3), x_data);
      x.inplace_multiply_const(.5);
      EXPECT_TRUE(vector_match(y2_data, x.to_vector()));
    }
    {
      Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 3), x_data);
      x *= .5;
      EXPECT_TRUE(vector_match(y2_data, x.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckCopyAndInplaceMultiplyConst) {
  const vector<float> x_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> y1_data {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24};
  const vector<float> y2_data {.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6};

  for (Device *dev : devices) {
    {
      Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 3), x_data);

      const Tensor copied = x;

      x.inplace_multiply_const(2);
      EXPECT_TRUE(vector_match(y1_data, x.to_vector()));
      EXPECT_TRUE(vector_match(x_data, copied.to_vector()));
    }
    {
      Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 3), x_data);

      const Tensor copied = x;

      x *= 2;
      EXPECT_TRUE(vector_match(y1_data, x.to_vector()));
      EXPECT_TRUE(vector_match(x_data, copied.to_vector()));
    }
    {
      Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 3), x_data);

      const Tensor copied = x;

      x.inplace_multiply_const(.5);
      EXPECT_TRUE(vector_match(y2_data, x.to_vector()));
      EXPECT_TRUE(vector_match(x_data, copied.to_vector()));
    }
    {
      Tensor x = dev->new_tensor_by_vector(Shape({2, 2}, 3), x_data);

      const Tensor copied = x;

      x *= .5;
      EXPECT_TRUE(vector_match(y2_data, x.to_vector()));
      EXPECT_TRUE(vector_match(x_data, copied.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckInplaceAddNN) {
  const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> b_data {0, -1, -2, -3, -3, -4, -5, -6, -6, -7, -8, -9};
  const vector<float> y_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};

  for (Device *dev : devices) {
    {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);
      a.inplace_add(b);
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
    {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);
      a += b;
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckInplaceAdd1N) {
  const vector<float> a_data {1, 2, 3, 4};
  const vector<float> b_data {0, -1, -2, -3, -3, -4, -5, -6, -6, -7, -8, -9};
  const vector<float> y_data {-8, -10, -12, -14};

  for (Device *dev : devices) {
    {
      Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
      const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);
      a.inplace_add(b);
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
    {
      Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
      const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);
      a += b;
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckInplaceAddN1) {
  const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> b_data {0, -1, -2, -3};
  const vector<float> y_data {1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9};

  for (Device *dev : devices) {
    {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector({2, 2}, b_data);
      a.inplace_add(b);
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
    {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector({2, 2}, b_data);
      a += b;
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckCopyAndInplaceAdd) {
  const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> b_data {0, -1, -2, -3, -3, -4, -5, -6, -6, -7, -8, -9};
  const vector<float> y_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};

  for (Device *dev : devices) {
    {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);

      const Tensor copied = a;

      a.inplace_add(b);
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
      EXPECT_TRUE(vector_match(a_data, copied.to_vector()));
    }
    {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);

      const Tensor copied = a;

      a += b;
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
      EXPECT_TRUE(vector_match(a_data, copied.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckInplaceSubtractNN) {
  const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> b_data {0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9};
  const vector<float> y_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};

  for (Device *dev : devices) {
    {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);
      a.inplace_subtract(b);
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
    {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);
      a -= b;
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckInplaceSubtract1N) {
  const vector<float> a_data {1, 2, 3, 4};
  const vector<float> b_data {0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9};
  const vector<float> y_data {-8, -10, -12, -14};
  for (Device *dev : devices) {
    {
      Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
      const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);
      a.inplace_subtract(b);
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
    {
      Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
      const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);
      a -= b;
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckInplaceSubtractN1) {
  const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> b_data {0, 1, 2, 3};
  const vector<float> y_data {1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9};

  for (Device *dev : devices) {
    {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector({2, 2}, b_data);
      a.inplace_subtract(b);
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
    {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector({2, 2}, b_data);
      a -= b;
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckCopyAndInplaceSubtract) {
  const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const vector<float> b_data {0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9};
  const vector<float> y_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};

  for (Device *dev : devices) {
    {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);

      const Tensor copied = a;

      a.inplace_subtract(b);
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
      EXPECT_TRUE(vector_match(a_data, copied.to_vector()));
    }
    {
      Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);

      const Tensor copied = a;

      a -= b;
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
      EXPECT_TRUE(vector_match(a_data, copied.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckInvalidInplaceOps) {
  const vector<Shape> shapes {
    Shape(),
    Shape({}, 3),
    Shape({2, 2}, 2),
  };

  for (Device *dev : devices) {
    Tensor a = dev->new_tensor_by_constant(Shape({2, 2}, 3), 0);

    for (const Shape &shape : shapes) {
      Tensor b = dev->new_tensor_by_constant(shape, 0);
      EXPECT_THROW(a.inplace_add(b), Error);
      EXPECT_THROW(a += b, Error);
      EXPECT_THROW(a.inplace_subtract(b), Error);
      EXPECT_THROW(a -= b, Error);
    }
  }
}

TEST_F(TensorTest, CheckArgMaxDims) {
  const vector<float> data = {
    0, 1, 2, 6, 7, 8, 3, 4, 5, -3, -4, -5, 0, -1, -2, -6, -7, -8,
  };
  const vector<vector<std::uint32_t>> expected = {
    {2, 2, 2, 0, 0, 0},
    {1, 1, 1, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  };

  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({3, 3}, 2), data);
    for (const std::uint32_t i : {0u, 1u, 2u}) {
      EXPECT_TRUE(vector_match(expected[i], a.argmax(i)));
    }
  }
}

TEST_F(TensorTest, CheckArgMaxLarge) {
  std::mt19937 rng;
  const vector<std::uint32_t> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024,
    1025, 2047, 2048, 2049, 65535, 65536, 65537,
  };

  for (Device *dev : devices) {
    for (const std::uint32_t n : ns) {
      if (n >= (1 << 11) && dev->type() == DeviceType::CUDA16) {
        // NOTE(odashi):
        // Half-precision types have only (10+1) bits resolution.
        continue;
      }

      vector<float> data(n);
      std::iota(begin(data), end(data), 0);
      std::shuffle(begin(data), end(data), rng);
      const auto it = std::find(begin(data), end(data), n - 1);
      const std::uint32_t pos = std::distance(begin(data), it);
      const Tensor a = dev->new_tensor_by_vector({n}, data);
      const vector<std::uint32_t> expected {pos};
      EXPECT_TRUE(vector_match(expected, a.argmax(0)));
    }
  }
}

TEST_F(TensorTest, CheckArgMaxMultipleLarge) {
  std::mt19937 rng;
  const vector<std::uint32_t> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024,
    1025, 2047, 2048, 2049, 65535, 65536, 65537,
  };

  for (Device *dev : devices) {
    for (const std::uint32_t n : ns) {
      if (n >= (1 << 11) && dev->type() == DeviceType::CUDA16) {
        // NOTE(vbkaisetsu):
        // Half-precision types have only (10+1) bits resolution.
        continue;
      }
      vector<float> data(n);
      std::iota(begin(data), end(data), 0);
      // NOTE(vbkaisetsu):
      // Generates a tensor that has some duplicated maximum values.
      for (std::uint32_t i = 0; i < 10 && i < n; ++i) {
        data[i] = n - 1;
      }
      std::shuffle(begin(data), end(data), rng);
      const auto it = std::find(begin(data), end(data), n - 1);
      const std::uint32_t pos = std::distance(begin(data), it);
      const Tensor a = dev->new_tensor_by_vector({n}, data);
      const vector<std::uint32_t> expected {pos};
      EXPECT_TRUE(vector_match(expected, a.argmax(0)));
    }
  }
}

TEST_F(TensorTest, CheckArgMinDims) {
  const vector<float> data = {
    3, 4, 5, 0, 1, 2, 6, 7, 8, 0, -1, -2, -6, -7, -8, -3, -4, -5,
  };
  const vector<vector<std::uint32_t>> expected = {
    {0, 0, 0, 2, 2, 2},
    {1, 1, 1, 1, 1, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  };

  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(Shape({3, 3}, 2), data);
    for (const std::uint32_t i : {0u, 1u, 2u}) {
      EXPECT_TRUE(vector_match(expected[i], a.argmin(i)));
    }
  }
}

TEST_F(TensorTest, CheckArgMinLarge) {
  std::mt19937 rng;
  const vector<std::uint32_t> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024,
    1025, 2047, 2048, 2049, 65535, 65536, 65537,
  };

  for (Device *dev : devices) {
    for (const std::uint32_t n : ns) {
      if (n >= (1 << 11) && dev->type() == DeviceType::CUDA16) {
        // NOTE(odashi):
        // Half-precision types have only (10+1) bits resolution.
        continue;
      }
      vector<float> data(n);
      std::iota(begin(data), end(data), 0);
      std::shuffle(begin(data), end(data), rng);
      const auto it = std::find(begin(data), end(data), 0);
      const std::uint32_t pos = std::distance(begin(data), it);
      const Tensor a = dev->new_tensor_by_vector({n}, data);
      const vector<std::uint32_t> expected {pos};
      EXPECT_TRUE(vector_match(expected, a.argmin(0)));
    }
  }
}

TEST_F(TensorTest, CheckArgMinMultipleLarge) {
  std::mt19937 rng;
  const vector<std::uint32_t> ns {
    1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024,
    1025, 2047, 2048, 2049, 65535, 65536, 65537,
  };

  for (Device *dev : devices) {
    for (const std::uint32_t n : ns) {
      if (n >= (1 << 11) && dev->type() == DeviceType::CUDA16) {
        // NOTE(vbkaisetsu):
        // Half-precision types have only (10+1) bits resolution.
        continue;
      }
      vector<float> data(n);
      std::iota(begin(data), end(data), 0);
      // NOTE(vbkaisetsu):
      // Generates a tensor that has some duplicated minimum values.
      for (std::uint32_t i = 0; i < 10 && i < n; ++i) {
        data[i] = 0;
      }
      std::shuffle(begin(data), end(data), rng);
      const auto it = std::find(begin(data), end(data), 0);
      const std::uint32_t pos = std::distance(begin(data), it);
      const Tensor a = dev->new_tensor_by_vector({n}, data);
      const vector<std::uint32_t> expected {pos};
      EXPECT_TRUE(vector_match(expected, a.argmin(0)));
    }
  }
}

}  // namespace primitiv
