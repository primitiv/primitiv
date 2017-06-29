#include <config.h>

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
using test_utils::vector_near;

namespace primitiv {

class TensorBackwardTest : public testing::Test {
protected:
  vector<Device *> devices;

  void SetUp() override {
    devices.emplace_back(new CPUDevice());
    devices.emplace_back(new CPUDevice()); // other device on the same hardware
#ifdef USE_CUDA
    devices.emplace_back(new CUDADevice(0));
    devices.emplace_back(new CUDADevice(0)); // other device on the same hardware
    if (CUDADevice::num_devices() > 2) {
      devices.emplace_back(new CUDADevice(1));
    }
#endif  // USE_CUDA
  }

  void TearDown() override {
    for (Device *dev : devices) {
      delete dev;
    }
  }
};

TEST_F(TensorBackwardTest, CheckMatMul11) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2, 2}, {1, 2, 3, 4});
    const Tensor b = dev->new_tensor_by_vector({2, 2}, {1, 0, 0, 2});
    const Tensor gy = dev->new_tensor_by_vector({2, 2}, {1, -1, 2, -2});
    Tensor ga = dev->new_tensor(a.shape(), 0);
    Tensor gb = dev->new_tensor(b.shape(), 0);
    dev->matmul_bw(a, b, gy, ga, gb);
    const vector<float> ga_val {1, -1, 4, -4};
    const vector<float> gb_val {-1, -1, -2, -2};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckMatMulNN) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {1, 2, 3, 4, -1, -2, -3, -4});
    const Tensor b = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {1, 0, 0, 2, 0, 1, 2, 0});
    const Tensor gy = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor ga = dev->new_tensor(a.shape(), 0);
    Tensor gb = dev->new_tensor(b.shape(), 0);
    dev->matmul_bw(a, b, gy, ga, gb);
    const vector<float> ga_val {1, -1, 4, -4, 2, -2, 2, -2};
    const vector<float> gb_val {-1, -1, -2, -2, 2, 2, 1, 1};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckMatMul1N) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector({2, 2}, {1, 2, 3, 4});
    const Tensor b = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {1, 0, 0, 2, 0, 1, 2, 0});
    const Tensor gy = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor ga = dev->new_tensor(a.shape(), 0);
    Tensor gb = dev->new_tensor(b.shape(), 0);
    dev->matmul_bw(a, b, gy, ga, gb);
    const vector<float> ga_val {3, -3, 6, -6};
    const vector<float> gb_val {-1, -1, -2, -2, -2, -2, -1, -1};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

TEST_F(TensorBackwardTest, CheckMatMulN1) {
  for (Device *dev : devices) {
    const Tensor a = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {1, 2, 3, 4, -1, -2, -3, -4});
    const Tensor b = dev->new_tensor_by_vector({2, 2}, {1, 0, 0, 2});
    const Tensor gy = dev->new_tensor_by_vector(
        Shape({2, 2}, 2), {1, -1, 2, -2, 2, -2, 1, -1});
    Tensor ga = dev->new_tensor(a.shape(), 0);
    Tensor gb = dev->new_tensor(b.shape(), 0);
    dev->matmul_bw(a, b, gy, ga, gb);
    const vector<float> ga_val {1, -1, 4, -4, 2, -2, 2, -2};
    const vector<float> gb_val {1, 1, -1, -1};
    EXPECT_TRUE(vector_match(ga_val, ga.to_vector()));
    EXPECT_TRUE(vector_match(gb_val, gb.to_vector()));
  }
}

}  // namespace primitiv
