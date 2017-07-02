#include <config.h>

#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/cpu_device.h>
#include <primitiv/error.h>
#include <primitiv/tensor.h>
#include <test_utils.h>

#ifdef USE_CUDA
#include <primitiv/cuda_device.h>
#endif  // USE_CUDA

using std::vector;
using test_utils::vector_match;

namespace primitiv {

class TensorTest : public testing::Test {
protected:
  vector<Device *> devices;

  void SetUp() override {
    devices.emplace_back(new CPUDevice());
#ifdef USE_CUDA
    devices.emplace_back(new CUDADevice(0));
#endif  // USE_CUDA
  }

  void TearDown() override {
    for (Device *dev : devices) {
      delete dev;
    }
  }
};

TEST_F(TensorTest, CheckNewDelete) {
  for (Device *dev : devices) {
    Tensor x1 = dev->new_tensor(Shape()); // 1 value
    Tensor x2 = dev->new_tensor(Shape {16, 16}); // 256 values
    Tensor x3 = dev->new_tensor(Shape({16, 16, 16}, 16)); // 65536 values
    // According to the C++ standard, local values are destroyed in the order:
    // x3 -> x2 -> x1 -> dev.
    // Then `dev` has no remaining memories.
  }
  SUCCEED();
}

// NOTE(odashi):
//   Now the CPUDevice does not manage memories itself, and the CUDADevice is
//   run on multi-threading that does not allow to perform the death test due to
//   the constraint of gTest.
#if 0
TEST_F(TensorTest, CheckInvalidNewDelete) {
  for (Device *dev : devices) {
    EXPECT_DEATH({
      Tensor x0;
      x0 = dev->new_tensor(Shape());
      // Local values are destroyed in the order: dev -> x0.
      // `x0` still have a memory when destroying `dev` and the process will
      // abort.
    }, "");
  }
}
#endif

TEST_F(TensorTest, CheckNewDefault) {
  const Tensor x;
  EXPECT_FALSE(x.valid());
  EXPECT_EQ(Shape(), x.shape());
  EXPECT_EQ(nullptr, x.device());
  EXPECT_EQ(nullptr, x.data());
}

TEST_F(TensorTest, CheckNewScalar) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor({});
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape(), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    const vector<float> d = x.to_vector();
    EXPECT_EQ(1u, d.size());
    EXPECT_EQ(dev, x.device());
  }
}

TEST_F(TensorTest, CheckNewMatrix) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape {2, 3});
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape({2, 3}), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    const vector<float> d = x.to_vector();
    EXPECT_EQ(6u, d.size());
    EXPECT_EQ(dev, x.device());
  }
}

TEST_F(TensorTest, CheckNewMatrixMinibatch) {
  for (Device *dev : devices) {
    const Tensor x = dev->new_tensor(Shape({2, 3}, 4));
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape({2, 3}, 4), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    const vector<float> d = x.to_vector();
    EXPECT_EQ(24u, d.size());
    EXPECT_EQ(dev, x.device());
  }
}

TEST_F(TensorTest, CheckNewScalarWithData) {
  for (Device *dev : devices ) {
    const Tensor x = dev->new_tensor({}, 1);
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape(), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    EXPECT_TRUE(vector_match(vector<float> {1}, x.to_vector()));
  }
}

TEST_F(TensorTest, CheckNewMatrixWithData) {
  for (Device *dev : devices ) {
    const vector<float> data {1, 2, 3, 4, 5, 6};
    const Tensor x = dev->new_tensor_by_vector({2, 3}, data);
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape({2, 3}), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    EXPECT_TRUE(vector_match(data, x.to_vector()));
  }
}

TEST_F(TensorTest, CheckNewMatrixMinibatchWithData) {
  for (Device *dev : devices ) {
    const vector<float> data {
      3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8,
      9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4,
    };
    const Tensor x = dev->new_tensor_by_vector(Shape({2, 3}, 4), data);
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape({2, 3}, 4), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    EXPECT_TRUE(vector_match(data, x.to_vector()));
  }
}

TEST_F(TensorTest, CheckMoveValidToNew) {
  for (Device *dev : devices) {
    Tensor tmp = dev->new_tensor_by_vector(Shape({2}, 3), {1, 2, 3, 4, 5, 6});
    void *ptr = tmp.data();

    const Tensor x = std::move(tmp);
    EXPECT_TRUE(x.valid());
    EXPECT_FALSE(tmp.valid());
    EXPECT_EQ(Shape({2}, 3), x.shape());
    EXPECT_EQ(ptr, x.data());
    EXPECT_TRUE(vector_match({1, 2, 3, 4, 5, 6}, x.to_vector()));
  }
}

TEST_F(TensorTest, CheckMoveValidToValid) {
  for (Device *dev : devices) {
    Tensor tmp = dev->new_tensor_by_vector({6}, {2, 4, 6, 8, 10 ,12});
    void *ptr = tmp.data();

    Tensor x = dev->new_tensor({}, 1);
    ASSERT_TRUE(x.valid());

    x = std::move(tmp);
    EXPECT_TRUE(x.valid());
    EXPECT_FALSE(tmp.valid());
    EXPECT_EQ(Shape({6}), x.shape());
    EXPECT_EQ(ptr, x.data());
    EXPECT_TRUE(vector_match({2, 4, 6, 8, 10 ,12}, x.to_vector()));
  }
}

TEST_F(TensorTest, CheckMoveValidToInvalid) {
  for (Device *dev : devices) {
    Tensor tmp = dev->new_tensor_by_vector(
        Shape({1}, 6), {3, 6, 9, 12, 15, 18});
    void *ptr = tmp.data();

    Tensor x;
    ASSERT_FALSE(x.valid());

    x = std::move(tmp);
    EXPECT_TRUE(x.valid());
    EXPECT_FALSE(tmp.valid());
    EXPECT_EQ(Shape({1}, 6), x.shape());
    EXPECT_EQ(ptr, x.data());
    EXPECT_TRUE(vector_match({3, 6, 9, 12, 15, 18}, x.to_vector()));
  }
}

TEST_F(TensorTest, CheckMoveValidToThis) {
  for (Device *dev : devices) {
    Tensor x = dev->new_tensor_by_vector({6}, {2, 4, 6, 8, 10 ,12});
    void *ptr = x.data();

    x = std::move(x);
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape({6}), x.shape());
    EXPECT_EQ(ptr, x.data());
    EXPECT_TRUE(vector_match({2, 4, 6, 8, 10 ,12}, x.to_vector()));
  }
}

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

    Tensor x = dev->new_tensor({}, 1);
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

TEST_F(TensorTest, CheckMoveInvalidToThis) {
  Tensor x;
  ASSERT_FALSE(x.valid());

  x = std::move(x);
  EXPECT_FALSE(x.valid());
}

TEST_F(TensorTest, CheckCopyValidToNew) {
  for (Device *dev : devices) {
    const Tensor tmp = dev->new_tensor_by_vector(Shape({2}, 3), {1, 2, 3, 4, 5, 6});
    const void *ptr = tmp.data();

    const Tensor x = tmp;
    EXPECT_TRUE(x.valid());
    EXPECT_TRUE(tmp.valid());
    EXPECT_EQ(Shape({2}, 3), x.shape());
    EXPECT_EQ(Shape({2}, 3), tmp.shape());
    EXPECT_EQ(ptr, x.data());
    EXPECT_EQ(ptr, tmp.data());
    EXPECT_TRUE(vector_match({1, 2, 3, 4, 5, 6}, x.to_vector()));
    EXPECT_TRUE(vector_match({1, 2, 3, 4, 5, 6}, tmp.to_vector()));
  }
}

TEST_F(TensorTest, CheckCopyValidToValid) {
  for (Device *dev : devices) {
    const Tensor tmp = dev->new_tensor_by_vector({6}, {2, 4, 6, 8, 10 ,12});
    const void *ptr = tmp.data();

    Tensor x = dev->new_tensor({}, 1);
    ASSERT_TRUE(x.valid());

    x = tmp;
    EXPECT_TRUE(x.valid());
    EXPECT_TRUE(tmp.valid());
    EXPECT_EQ(Shape({6}), x.shape());
    EXPECT_EQ(Shape({6}), tmp.shape());
    EXPECT_EQ(ptr, static_cast<const Tensor>(x).data());
    EXPECT_EQ(ptr, tmp.data());
    EXPECT_TRUE(vector_match({2, 4, 6, 8, 10 ,12}, x.to_vector()));
    EXPECT_TRUE(vector_match({2, 4, 6, 8, 10 ,12}, tmp.to_vector()));
  }
}

TEST_F(TensorTest, CheckCopyValidToInvalid) {
  for (Device *dev : devices) {
    const Tensor tmp = dev->new_tensor_by_vector(
        Shape({1}, 6), {3, 6, 9, 12, 15, 18});
    const void *ptr = tmp.data();

    Tensor x;
    ASSERT_FALSE(x.valid());

    x = tmp;
    EXPECT_TRUE(x.valid());
    EXPECT_TRUE(tmp.valid());
    EXPECT_EQ(Shape({1}, 6), x.shape());
    EXPECT_EQ(Shape({1}, 6), tmp.shape());
    EXPECT_EQ(ptr, static_cast<const Tensor>(x).data());
    EXPECT_EQ(ptr, tmp.data());
    EXPECT_TRUE(vector_match({3, 6, 9, 12, 15, 18}, x.to_vector()));
    EXPECT_TRUE(vector_match({3, 6, 9, 12, 15, 18}, tmp.to_vector()));
  }
}

TEST_F(TensorTest, CheckCopyValidToThis) {
  for (Device *dev : devices) {
    Tensor x = dev->new_tensor_by_vector({6}, {2, 4, 6, 8, 10 ,12});
    void *ptr = x.data();

    x = x;
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape({6}), x.shape());
    EXPECT_EQ(ptr, x.data());
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

    Tensor x = dev->new_tensor({}, 1);
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

TEST_F(TensorTest, CheckUnique) {
  for (Device *dev : devices) {
    Tensor x = dev->new_tensor({});
    const void *ptr11 = static_cast<const Tensor>(x).data();
    void *ptr12 = x.data();
    const void *ptr13 = static_cast<const Tensor>(x).data();
    void *ptr14 = x.data();
    EXPECT_EQ(ptr12, ptr11);
    EXPECT_EQ(ptr13, ptr11);
    EXPECT_EQ(ptr14, ptr11);

    const Tensor copied = x;
    const void *ptr21 = static_cast<const Tensor>(x).data();
    void *ptr22 = x.data();
    const void *ptr23 = static_cast<const Tensor>(x).data();
    void *ptr24 = x.data();
    EXPECT_EQ(ptr21, ptr11);
    EXPECT_NE(ptr22, ptr11);
    EXPECT_NE(ptr23, ptr11);
    EXPECT_NE(ptr24, ptr11);
    EXPECT_EQ(ptr23, ptr22);
    EXPECT_EQ(ptr24, ptr22);
  }
}

TEST_F(TensorTest, CheckResetValuesByConstant) {
  for (Device *dev : devices) {
    {
      const Tensor x = dev->new_tensor(Shape({2, 2}, 2), 42);
      EXPECT_TRUE(vector_match(vector<float>(8, 42), x.to_vector()));
    }
    {
      Tensor x = dev->new_tensor(Shape({2, 2}, 2));
      const void *ptr = x.data();
      x.reset(42);
      EXPECT_EQ(ptr, x.data());
      EXPECT_TRUE(vector_match(vector<float>(8, 42), x.to_vector()));
    }
    {
      Tensor x = dev->new_tensor(Shape({2, 2}, 2), 123);
      const Tensor copied = x;
      EXPECT_EQ(static_cast<const Tensor>(x).data(), copied.data());

      x.reset(42);
      EXPECT_NE(static_cast<const Tensor>(x).data(), copied.data());
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
      Tensor x = dev->new_tensor(Shape({2, 2}, 2));
      const void *ptr = x.data();
      x.reset_by_array(data);
      EXPECT_EQ(ptr, x.data());
      EXPECT_TRUE(
          vector_match(vector<float> {1, 2, 3, 4, 5, 6, 7, 8}, x.to_vector()));
    }
    {
      const float data[] {1, 2, 3, 4, 5, 6, 7, 8};
      Tensor x = dev->new_tensor(Shape({2, 2}, 2), 123);
      const Tensor copied = x;
      EXPECT_EQ(static_cast<const Tensor>(x).data(), copied.data());

      x.reset_by_array(data);
      EXPECT_NE(static_cast<const Tensor>(x).data(), copied.data());
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
      Tensor x = dev->new_tensor(Shape({2, 2}, 2));
      const void *ptr = x.data();
      x.reset_by_vector(data);
      EXPECT_EQ(ptr, x.data());
      EXPECT_TRUE(vector_match(data, x.to_vector()));
    }
    {
      const vector<float> data {1, 2, 3, 4, 5, 6, 7, 8};
      Tensor x = dev->new_tensor(Shape({2, 2}, 2), 123);
      const Tensor copied = x;
      EXPECT_EQ(static_cast<const Tensor>(x).data(), copied.data());

      x.reset_by_vector(data);
      EXPECT_NE(static_cast<const Tensor>(x).data(), copied.data());
      EXPECT_TRUE(vector_match(data, x.to_vector()));
      EXPECT_TRUE(vector_match(vector<float>(8, 123), copied.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckAddGradientNN) {
  for (Device *dev : devices) {
    const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const vector<float> b_data {0, -1, -2, -3, -3, -4, -5, -6, -6, -7, -8, -9};
    const vector<float> y_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);
    dev->add_gradient(b, a);
    EXPECT_TRUE(vector_match(y_data, a.to_vector()));
  }
}

TEST_F(TensorTest, CheckAddGradient1N) {
  for (Device *dev : devices) {
    const vector<float> a_data {1, 2, 3, 4};
    const vector<float> b_data {0, -1, -2, -3, -3, -4, -5, -6, -6, -7, -8, -9};
    const vector<float> y_data {-8, -10, -12, -14};
    Tensor a = dev->new_tensor_by_vector({2, 2}, a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);
    dev->add_gradient(b, a);
    EXPECT_TRUE(vector_match(y_data, a.to_vector()));
  }
}

TEST_F(TensorTest, CheckAddGradientN1) {
  for (Device *dev : devices) {
    const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const vector<float> b_data {0, -1, -2, -3};
    const vector<float> y_data {1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9};
    Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
    const Tensor b = dev->new_tensor_by_vector({2, 2}, b_data);
    dev->add_gradient(b, a);
    EXPECT_TRUE(vector_match(y_data, a.to_vector()));
  }
}

TEST_F(TensorTest, CheckInvalidAddGradient) {
  for (Device *dev : devices) {
    vector<Shape> shapes {
      Shape(),
      Shape({}, 3),
      Shape({2, 2}, 2),
    };
    Tensor a = dev->new_tensor(Shape({2, 2}, 3));

    for (const Shape &shape : shapes) {
      Tensor b = dev->new_tensor(shape);
      EXPECT_THROW(dev->add_gradient(b, a), Error);
    }
  }
}

TEST_F(TensorTest, CheckCopyAndAddGradient) {
  for (Device *dev : devices) {
    const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const vector<float> b_data {0, -1, -2, -3, -3, -4, -5, -6, -6, -7, -8, -9};
    const vector<float> y_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    Tensor a = dev->new_tensor_by_vector(Shape({2, 2}, 3), a_data);
    const Tensor b = dev->new_tensor_by_vector(Shape({2, 2}, 3), b_data);

    const Tensor copied = a;
    EXPECT_EQ(static_cast<const Tensor>(a).data(), copied.data());

    dev->add_gradient(b, a);
    EXPECT_NE(static_cast<const Tensor>(a).data(), copied.data());
    EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    EXPECT_TRUE(vector_match(a_data, copied.to_vector()));
  }
}

}  // namespace primitiv
