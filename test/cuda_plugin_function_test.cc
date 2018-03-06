#include <primitiv/config.h>
#include <dlls/location.h>

#include <vector>
#include <gtest/gtest.h>
#include <primitiv/error.h>
#include <primitiv/functions.h>
#include <primitiv/plugin_function.h>
#include <test_utils.h>

#include <primitiv/cuda_device.h>
#include <primitiv/naive_device.h>

using std::vector;
using test_utils::vector_match;

namespace primitiv {

class CUDAPluginFunctionTest : public testing::Test {};

TEST_F(CUDAPluginFunctionTest, CheckInitialize) {
  const std::string dirname = DLLS_DIR;
  PluginFunction pf(dirname + "/test_plugin_function.cudll");

  EXPECT_EQ(2u, pf.num_arguments());
  EXPECT_EQ(1u, pf.num_returns());
}

TEST_F(CUDAPluginFunctionTest, CheckForwardShape) {
  const std::string dirname = DLLS_DIR;
  PluginFunction pf(dirname + "/test_plugin_function.cudll");

  {
    Shape sa({3}, 2);
    Shape sb({3}, 2);
    Shape sy;
    EXPECT_NO_THROW(pf.forward_shape({ &sa, &sb }, { &sy }));
    EXPECT_EQ(Shape({3}, 2), sy);
  }
  {
    Shape sa {3};
    Shape sb({3}, 2);
    Shape sy;
    EXPECT_NO_THROW(pf.forward_shape({ &sa, &sb }, { &sy }));
    EXPECT_EQ(Shape({3}, 2), sy);
  }
  {
    Shape sa({3}, 2);
    Shape sb {3};
    Shape sy;
    EXPECT_NO_THROW(pf.forward_shape({ &sa, &sb }, { &sy }));
    EXPECT_EQ(Shape({3}, 2), sy);
  }
}

TEST_F(CUDAPluginFunctionTest, CheckForward) {
  const std::string dirname = DLLS_DIR;
  PluginFunction pf(dirname + "/test_plugin_function.cudll");

  devices::CUDA dev(0);
  Device::set_default(dev);

  {
    const Tensor a = functions::input<Tensor>(
        Shape({3}, 2), {1, 2, 3, 4, 5, 6});
    const Tensor b = functions::ones<Tensor>(Shape({3}, 2));
    Tensor y = dev.new_raw_tensor(Shape({3}, 2));
    EXPECT_NO_THROW(pf.forward({ &a, &b }, { &y }));
    EXPECT_TRUE(vector_match(vector<float> {2, 3, 4, 5, 6, 7}, y.to_vector()));
  }
  {
    const Tensor a = functions::input<Tensor>(
        Shape({3}, 2), {1, 2, 3, 4, 5, 6});
    const Tensor b = functions::ones<Tensor>({3});
    Tensor y = dev.new_raw_tensor(Shape({3}, 2));
    EXPECT_NO_THROW(pf.forward({ &a, &b }, { &y }));
    EXPECT_TRUE(vector_match(vector<float> {2, 3, 4, 5, 6, 7}, y.to_vector()));
  }
  {
    const Tensor a = functions::input<Tensor>({3}, {1, 2, 3});
    const Tensor b = functions::ones<Tensor>(Shape({3}, 2));
    Tensor y = dev.new_raw_tensor(Shape({3}, 2));
    EXPECT_NO_THROW(pf.forward({ &a, &b }, { &y }));
    EXPECT_TRUE(vector_match(vector<float> {2, 3, 4, 2, 3, 4}, y.to_vector()));
  }
}

TEST_F(CUDAPluginFunctionTest, CheckBackward) {
  const std::string dirname = DLLS_DIR;
  PluginFunction pf(dirname + "/test_plugin_function.cudll");

  devices::CUDA dev(0);
  Device::set_default(dev);

  {
    const Tensor a = functions::zeros<Tensor>(Shape({3}, 2));
    const Tensor b = functions::zeros<Tensor>(Shape({3}, 2));
    const Tensor y = functions::zeros<Tensor>(Shape({3}, 2));
    const Tensor gy = functions::ones<Tensor>(Shape({3}, 2));
    Tensor ga = functions::ones<Tensor>(Shape({3}, 2));
    Tensor gb = functions::ones<Tensor>(Shape({3}, 2));
    EXPECT_NO_THROW(pf.backward({ &a, &b }, { &y }, { &gy }, { &ga, &gb }));
    EXPECT_TRUE(vector_match(vector<float>(6, 2), ga.to_vector()));
    EXPECT_TRUE(vector_match(vector<float>(6, 2), gb.to_vector()));
  }
  {
    const Tensor a = functions::zeros<Tensor>({3});
    const Tensor b = functions::zeros<Tensor>(Shape({3}, 2));
    const Tensor y = functions::zeros<Tensor>(Shape({3}, 2));
    const Tensor gy = functions::ones<Tensor>(Shape({3}, 2));
    Tensor ga = functions::ones<Tensor>({3});
    Tensor gb = functions::ones<Tensor>(Shape({3}, 2));
    EXPECT_NO_THROW(pf.backward({ &a, &b }, { &y }, { &gy }, { &ga, &gb }));
    EXPECT_TRUE(vector_match(vector<float>(3, 3), ga.to_vector()));
    EXPECT_TRUE(vector_match(vector<float>(6, 2), gb.to_vector()));
  }
  {
    const Tensor a = functions::zeros<Tensor>(Shape({3}, 2));
    const Tensor b = functions::zeros<Tensor>({3});
    const Tensor y = functions::zeros<Tensor>(Shape({3}, 2));
    const Tensor gy = functions::ones<Tensor>(Shape({3}, 2));
    Tensor ga = functions::ones<Tensor>(Shape({3}, 2));
    Tensor gb = functions::ones<Tensor>({3});
    EXPECT_NO_THROW(pf.backward({ &a, &b }, { &y }, { &gy }, { &ga, &gb }));
    EXPECT_TRUE(vector_match(vector<float>(6, 2), ga.to_vector()));
    EXPECT_TRUE(vector_match(vector<float>(3, 3), gb.to_vector()));
  }
}

TEST_F(CUDAPluginFunctionTest, CheckInvalidInitialize) {
  const std::string dirname = DLLS_DIR;
  EXPECT_THROW(PluginFunction pf(dirname + "/foo"), Error);
}

TEST_F(CUDAPluginFunctionTest, CheckInvalidOperations) {
  const std::string dirname = DLLS_DIR;
  PluginFunction pf(dirname + "/test_plugin_function.cudll");

  Shape sa {2};
  Shape sb {3};
  Shape sy;
  EXPECT_THROW(pf.forward_shape({ &sa, &sb }, { &sy }), Error);

  devices::Naive dev;
  Device::set_default(dev);
  const Tensor a = functions::zeros<Tensor>({});
  const Tensor b = functions::zeros<Tensor>({});
  const Tensor gy = functions::zeros<Tensor>({});
  Tensor y, ga, gb;
  EXPECT_THROW(pf.forward({ &a, &b }, { &y }), Error);
  EXPECT_THROW(pf.backward({ &a, &b }, { &y }, { &gy }, { &ga, &gb }), Error);
}

}  // namespace primitiv
