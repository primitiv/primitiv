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
    const Tensor x = dev->new_tensor({2, 3}, data);
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
    const Tensor x = dev->new_tensor(Shape({2, 3}, 4), data);
    EXPECT_TRUE(x.valid());
    EXPECT_EQ(Shape({2, 3}, 4), x.shape());
    EXPECT_NE(nullptr, const_cast<Tensor &>(x).data());
    EXPECT_TRUE(vector_match(data, x.to_vector()));
  }
}

TEST_F(TensorTest, CheckMoveCtor) {
  for (Device *dev : devices) {
    Tensor tmp = dev->new_tensor(Shape({2}, 3), {1, 2, 3, 4, 5, 6});
    void *ptr = tmp.data();

    Tensor x = std::move(tmp);

    EXPECT_TRUE(x.valid());
    EXPECT_FALSE(tmp.valid());
    EXPECT_EQ(Shape({2}, 3), x.shape());
    EXPECT_EQ(ptr, x.data());
    EXPECT_TRUE(vector_match({1, 2, 3, 4, 5, 6}, x.to_vector()));
  }
}

TEST_F(TensorTest, CheckMoveOpToValidObj) {
  for (Device *dev : devices) {
    Tensor tmp = dev->new_tensor({6}, {2, 4, 6, 8, 10 ,12});
    void *ptr = tmp.data();

    Tensor x = dev->new_tensor({}, 1);
    x = std::move(tmp);

    EXPECT_TRUE(x.valid());
    EXPECT_FALSE(tmp.valid());
    EXPECT_EQ(Shape({6}), x.shape());
    EXPECT_EQ(ptr, x.data());
    EXPECT_TRUE(vector_match({2, 4, 6, 8, 10 ,12}, x.to_vector()));
  }
}

TEST_F(TensorTest, CheckMoveOpToInvalidObj) {
  for (Device *dev : devices) {
    Tensor tmp = dev->new_tensor(Shape({1}, 6), {3, 6, 9, 12, 15, 18});
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

TEST_F(TensorTest, CheckAddGradientNN) {
  for (Device *dev : devices) {
    const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const vector<float> b_data {0, -1, -2, -3, -3, -4, -5, -6, -6, -7, -8, -9};
    const vector<float> y_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    Tensor a = dev->new_tensor(Shape({2, 2}, 3), a_data);
    const Tensor b = dev->new_tensor(Shape({2, 2}, 3), b_data);
    a.add_gradient(b);
    EXPECT_TRUE(vector_match(y_data, a.to_vector()));
  }
}

TEST_F(TensorTest, CheckAddGradient1N) {
  for (Device *dev : devices) {
    const vector<float> a_data {1, 2, 3, 4};
    const vector<float> b_data {0, -1, -2, -3, -3, -4, -5, -6, -6, -7, -8, -9};
    const vector<float> y_data {-8, -10, -12, -14};
    Tensor a = dev->new_tensor({2, 2}, a_data);
    const Tensor b = dev->new_tensor(Shape({2, 2}, 3), b_data);
    a.add_gradient(b);
    EXPECT_TRUE(vector_match(y_data, a.to_vector()));
  }
}

TEST_F(TensorTest, CheckAddGradientN1) {
  for (Device *dev : devices) {
    const vector<float> a_data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const vector<float> b_data {0, -1, -2, -3};
    const vector<float> y_data {1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9};
    Tensor a = dev->new_tensor(Shape({2, 2}, 3), a_data);
    const Tensor b = dev->new_tensor({2, 2}, b_data);
    a.add_gradient(b);
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
      EXPECT_THROW(a.add_gradient(b), Error);
    }
  }
}

TEST_F(TensorTest, CheckAddGradientOffsetNN_1) {
  const vector<float> a_data {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  const vector<float> b_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  const vector<float> y_data {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
  for (Device *dev : devices) {
    for (unsigned i : {0, 1, 2, 5, 10}) {
      Tensor a = dev->new_tensor(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor(Shape({2, 2}, 3), b_data);
      a.add_gradient_offset(b, i, 0);
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckAddGradientOffsetNN_2) {
  const vector<float> a_data {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  const vector<float> b_data {1, 1, 2, 2, 3, 3};
  struct TestCase {
    Shape shape;
    unsigned dim, offset;
    vector<float> y_data;
  };
  vector<TestCase> test_cases {
    {Shape({1, 2}, 3), 0, 0, {1, 1, 3, 3, 2, 1, 4, 3, 3, 1, 5, 3}},
    {Shape({1, 2}, 3), 0, 1, {0, 2, 2, 4, 0, 3, 2, 5, 0, 4, 2, 6}},
    {Shape({2}, 3), 1, 0, {1, 2, 2, 3, 2, 3, 2, 3, 3, 4, 2, 3}},
    {Shape({2}, 3), 1, 1, {0, 1, 3, 4, 0, 1, 4, 5, 0, 1, 5, 6}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor(tc.shape, b_data);
      a.add_gradient_offset(b, tc.dim, tc.offset);
      EXPECT_TRUE(vector_match(tc.y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckAddGradientOffset1N_1) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
  const vector<float> y_data {6, 7, 8, 9};
  for (Device *dev : devices) {
    for (unsigned i : {0, 1, 2, 5, 10}) {
      Tensor a = dev->new_tensor({2, 2}, a_data);
      const Tensor b = dev->new_tensor(Shape({2, 2}, 3), b_data);
      a.add_gradient_offset(b, i, 0);
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckAddGradientOffset1N_2) {
  const vector<float> a_data {0, 1, 2, 3};
  const vector<float> b_data {1, 1, 2, 2, 3, 3};
  struct TestCase {
    Shape shape;
    unsigned dim, offset;
    vector<float> y_data;
  };
  vector<TestCase> test_cases {
    {Shape({1, 2}, 3), 0, 0, {6, 1, 8, 3}},
    {Shape({1, 2}, 3), 0, 1, {0, 7, 2, 9}},
    {Shape({2}, 3), 1, 0, {6, 7, 2, 3}},
    {Shape({2}, 3), 1, 1, {0, 1, 8, 9}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor({2, 2}, a_data);
      const Tensor b = dev->new_tensor(tc.shape, b_data);
      a.add_gradient_offset(b, tc.dim, tc.offset);
      EXPECT_TRUE(vector_match(tc.y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckAddGradientOffsetN1_1) {
  const vector<float> a_data {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
  const vector<float> b_data {-1, -2, -3, -4};
  const vector<float> y_data {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  for (Device *dev : devices) {
    for (unsigned i : {0, 1, 2, 5, 10}) {
      Tensor a = dev->new_tensor(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor({2, 2}, b_data);
      a.add_gradient_offset(b, i, 0);
      EXPECT_TRUE(vector_match(y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckAddGradientOffsetN1_2) {
  const vector<float> a_data {1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
  const vector<float> b_data {-1, -2};
  struct TestCase {
    Shape shape;
    unsigned dim, offset;
    vector<float> y_data;
  };
  vector<TestCase> test_cases {
    {{1, 2}, 0, 0, {0, 2, 1, 4, 1, 3, 2, 5, 2, 4, 3, 6}},
    {{1, 2}, 0, 1, {1, 1, 3, 2, 2, 2, 4, 3, 3, 3, 5, 4}},
    {{2}, 1, 0, {0, 0, 3, 4, 1, 1, 4, 5, 2, 2, 5, 6}},
    {{2}, 1, 1, {1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4}},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor(Shape({2, 2}, 3), a_data);
      const Tensor b = dev->new_tensor(tc.shape, b_data);
      a.add_gradient_offset(b, tc.dim, tc.offset);
      EXPECT_TRUE(vector_match(tc.y_data, a.to_vector()));
    }
  }
}

TEST_F(TensorTest, CheckInvalidAddGradientOffset) {
  struct TestCase {
    Shape a_shape, b_shape;
    unsigned dim, offset;
    bool ok;
  };
  vector<TestCase> test_cases {
    {Shape({}, 2), Shape({}, 3), 0, 0},
    {{}, {}, 0, 0, true},
    {{}, {}, 0, 1, false},
    {{42}, {}, 0, 41, true},
    {{42}, {}, 0, 42, false},
    {{42}, {42}, 0, 0, true},
    {{42}, {42}, 0, 1, false},
    {{42}, {43}, 0, 0, false},
    {{42}, {4, 42}, 0, 0, false},
    {{42}, {4, 2, 42}, 0, 0, false},
    {{4, 4}, {2, 2}, 0, 0, false},
    {{4, 4}, {2, 4}, 0, 2, true},
    {{4, 4}, {2, 4}, 0, 3, false},
    {{4, 4}, {4, 2}, 1, 2, true},
    {{4, 4}, {4, 2}, 1, 3, false},
    {{4, 4}, {4, 4}, 2, 0, true},
    {{4, 4}, {4, 4}, 2, 1, false},
  };
  for (Device *dev : devices) {
    for (const TestCase &tc : test_cases) {
      Tensor a = dev->new_tensor(tc.a_shape, 0);
      const Tensor b = dev->new_tensor(tc.b_shape, 0);
      if (tc.ok) EXPECT_NO_THROW(a.add_gradient_offset(b, tc.dim, tc.offset));
      else EXPECT_THROW(a.add_gradient_offset(b, tc.dim, tc.offset), Error);
    }
  }
}

}  // namespace primitiv
