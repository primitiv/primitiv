#include <primitiv/config.h>

#include <cstdio>

#include <gtest/gtest.h>

#include <primitiv/core/arithmetic.h>
#include <primitiv/core/error.h>
#include <primitiv/devices/naive/device.h>
#include <primitiv/core/parameter.h>
#include <primitiv/core/optimizer_impl.h>

#include <test_utils.h>

using std::vector;
using test_utils::vector_match;
using test_utils::vector_near;

namespace primitiv {
namespace optimizers {

class OptimizerImplTest : public testing::Test {
protected:
  devices::Naive dev;
};

TEST_F(OptimizerImplTest, CheckDefaultHyperparameters) {
  SGD sgd;
  EXPECT_FLOAT_EQ(.1, sgd.eta());

  MomentumSGD momentumsgd;
  EXPECT_FLOAT_EQ(.01, momentumsgd.eta());
  EXPECT_FLOAT_EQ(.9, momentumsgd.momentum());

  AdaGrad adagrad;
  EXPECT_FLOAT_EQ(.001, adagrad.eta());
  EXPECT_FLOAT_EQ(1e-8, adagrad.eps());

  RMSProp rmsprop;
  EXPECT_FLOAT_EQ(.01, rmsprop.eta());
  EXPECT_FLOAT_EQ(.9, rmsprop.alpha());
  EXPECT_FLOAT_EQ(1e-8, rmsprop.eps());

  AdaDelta adadelta;
  EXPECT_FLOAT_EQ(.95, adadelta.rho());
  EXPECT_FLOAT_EQ(1e-6, adadelta.eps());

  Adam adam;
  EXPECT_FLOAT_EQ(.001, adam.alpha());
  EXPECT_FLOAT_EQ(.9, adam.beta1());
  EXPECT_FLOAT_EQ(.999, adam.beta2());
  EXPECT_FLOAT_EQ(1e-8, adam.eps());
}

TEST_F(OptimizerImplTest, CheckGivenHyperparameters) {
  SGD sgd(1);
  EXPECT_FLOAT_EQ(1, sgd.eta());

  MomentumSGD momentumsgd(1, 2);
  EXPECT_FLOAT_EQ(1, momentumsgd.eta());
  EXPECT_FLOAT_EQ(2, momentumsgd.momentum());

  AdaGrad adagrad(1, 2);
  EXPECT_FLOAT_EQ(1, adagrad.eta());
  EXPECT_FLOAT_EQ(2, adagrad.eps());

  RMSProp rmsprop(1, 2, 3);
  EXPECT_FLOAT_EQ(1, rmsprop.eta());
  EXPECT_FLOAT_EQ(2, rmsprop.alpha());
  EXPECT_FLOAT_EQ(3, rmsprop.eps());

  AdaDelta adadelta(1, 2);
  EXPECT_FLOAT_EQ(1, adadelta.rho());
  EXPECT_FLOAT_EQ(2, adadelta.eps());

  Adam adam(1, 2, 3, 4);
  EXPECT_FLOAT_EQ(1, adam.alpha());
  EXPECT_FLOAT_EQ(2, adam.beta1());
  EXPECT_FLOAT_EQ(3, adam.beta2());
  EXPECT_FLOAT_EQ(4, adam.eps());
}

TEST_F(OptimizerImplTest, CheckInvalidLoad) {
  SGD sgd;

  const std::string path = "/tmp/primitiv_OptimizerImplTest_CheckInvalidLoad.data";
  sgd.save(path);

  // NOTE(odashi):
  // Below function calls work successfully, but do not update optimizer
  // specific configurations that is not stored in the file.

  MomentumSGD momentumsgd;
  EXPECT_NO_THROW(momentumsgd.load(path));

  AdaGrad adagrad;
  EXPECT_NO_THROW(adagrad.load(path));

  RMSProp rmsprop;
  EXPECT_NO_THROW(rmsprop.load(path));

  AdaDelta adadelta;
  EXPECT_NO_THROW(adadelta.load(path));

  Adam adam;
  EXPECT_NO_THROW(adam.load(path));

  std::remove(path.c_str());
}

TEST_F(OptimizerImplTest, CheckSGDSaveLoad) {
  SGD optimizer(1);
  optimizer.set_epoch(2);
  optimizer.set_learning_rate_scaling(3);
  optimizer.set_weight_decay(4);
  optimizer.set_gradient_clipping(5);

  const std::string path = "/tmp/primitiv_OptimizerImplTest_CheckSGDSaveLoad.data";
  optimizer.save(path);

  SGD optimizer2;
  optimizer2.load(path);
  std::remove(path.c_str());

  EXPECT_EQ(1, optimizer2.eta());
  EXPECT_EQ(2u, optimizer2.get_epoch());
  EXPECT_EQ(3, optimizer2.get_learning_rate_scaling());
  EXPECT_EQ(4, optimizer2.get_weight_decay());
  EXPECT_EQ(5, optimizer2.get_gradient_clipping());
}

TEST_F(OptimizerImplTest, CheckSGDGetConfigs) {
  SGD optimizer(1);
  optimizer.set_epoch(2);
  optimizer.set_learning_rate_scaling(3);
  optimizer.set_weight_decay(4);
  optimizer.set_gradient_clipping(5);

  std::unordered_map<std::string, std::uint32_t> uint_configs;
  std::unordered_map<std::string, float> float_configs;
  optimizer.get_configs(uint_configs, float_configs);

  EXPECT_EQ(1u, uint_configs.size());
  EXPECT_EQ(4u, float_configs.size());
  EXPECT_EQ(1, float_configs.at("SGD.eta"));
  EXPECT_EQ(2u, uint_configs.at("Optimizer.epoch"));
  EXPECT_EQ(3, float_configs.at("Optimizer.lr_scale"));
  EXPECT_EQ(4, float_configs.at("Optimizer.l2_strength"));
  EXPECT_EQ(5, float_configs.at("Optimizer.clip_threshold"));
}

TEST_F(OptimizerImplTest, CheckSGDSetConfigs) {
  SGD optimizer(0);
  optimizer.set_epoch(0);
  optimizer.set_learning_rate_scaling(0);
  optimizer.set_weight_decay(0);
  optimizer.set_gradient_clipping(0);

  std::unordered_map<std::string, std::uint32_t> uint_configs {
    std::make_pair("Optimizer.epoch", 2),
  };
  std::unordered_map<std::string, float> float_configs {
    std::make_pair("SGD.eta", 1),
    std::make_pair("Optimizer.lr_scale", 3),
    std::make_pair("Optimizer.l2_strength", 4),
    std::make_pair("Optimizer.clip_threshold", 5),
  };
  optimizer.set_configs(uint_configs, float_configs);

  EXPECT_EQ(1, optimizer.eta());
  EXPECT_EQ(2u, optimizer.get_epoch());
  EXPECT_EQ(3, optimizer.get_learning_rate_scaling());
  EXPECT_EQ(4, optimizer.get_weight_decay());
  EXPECT_EQ(5, optimizer.get_gradient_clipping());
}

TEST_F(OptimizerImplTest, CheckSGDUpdate) {
  Parameter param({2, 2}, {1, 2, 3, 4}, dev);
  ASSERT_TRUE(vector_match(
        vector<float> {1, 2, 3, 4}, param.value().to_vector()));

  SGD optimizer;
  optimizer.set_learning_rate_scaling(.1);
  optimizer.add(param);

  vector<vector<float>> expected_v {
    {9.9000000e-01, 1.9800000e+00, 2.9700000e+00, 3.9600000e+00},
    {9.8010000e-01, 1.9602000e+00, 2.9403000e+00, 3.9204000e+00},
    {9.7029900e-01, 1.9405980e+00, 2.9108970e+00, 3.8811960e+00},
    {9.6059601e-01, 1.9211920e+00, 2.8817880e+00, 3.8423840e+00},
    {9.5099005e-01, 1.9019801e+00, 2.8529701e+00, 3.8039602e+00},
  };

  for (std::uint32_t i = 0; i < 5; ++i) {
    optimizer.reset_gradients();
    EXPECT_TRUE(vector_match(
          vector<float>(4, 0), param.gradient().to_vector()));

    param.gradient() += param.value();  // Squared loss
    optimizer.update();
    EXPECT_TRUE(vector_match(expected_v[i], param.value().to_vector()));
  }
}

TEST_F(OptimizerImplTest, CheckMomentumSGDSaveLoad) {
  MomentumSGD optimizer(1, 2);
  optimizer.set_epoch(3);
  optimizer.set_learning_rate_scaling(4);
  optimizer.set_weight_decay(5);
  optimizer.set_gradient_clipping(6);

  const std::string path = "/tmp/primitiv_OptimizerImplTest_CheckMomentumSGDSaveLoad.data";
  optimizer.save(path);

  MomentumSGD optimizer2;
  optimizer2.load(path);
  std::remove(path.c_str());

  EXPECT_EQ(1, optimizer2.eta());
  EXPECT_EQ(2, optimizer2.momentum());
  EXPECT_EQ(3u, optimizer2.get_epoch());
  EXPECT_EQ(4, optimizer2.get_learning_rate_scaling());
  EXPECT_EQ(5, optimizer2.get_weight_decay());
  EXPECT_EQ(6, optimizer2.get_gradient_clipping());
}

TEST_F(OptimizerImplTest, CheckMomentumSGDGetConfigs) {
  MomentumSGD optimizer(1, 2);
  optimizer.set_epoch(3);
  optimizer.set_learning_rate_scaling(4);
  optimizer.set_weight_decay(5);
  optimizer.set_gradient_clipping(6);

  std::unordered_map<std::string, std::uint32_t> uint_configs;
  std::unordered_map<std::string, float> float_configs;
  optimizer.get_configs(uint_configs, float_configs);

  EXPECT_EQ(1u, uint_configs.size());
  EXPECT_EQ(5u, float_configs.size());
  EXPECT_EQ(1, float_configs.at("MomentumSGD.eta"));
  EXPECT_EQ(2, float_configs.at("MomentumSGD.momentum"));
  EXPECT_EQ(3u, uint_configs.at("Optimizer.epoch"));
  EXPECT_EQ(4, float_configs.at("Optimizer.lr_scale"));
  EXPECT_EQ(5, float_configs.at("Optimizer.l2_strength"));
  EXPECT_EQ(6, float_configs.at("Optimizer.clip_threshold"));
}

TEST_F(OptimizerImplTest, CheckMomentumSGDSetConfigs) {
  MomentumSGD optimizer(0, 0);
  optimizer.set_epoch(0);
  optimizer.set_learning_rate_scaling(0);
  optimizer.set_weight_decay(0);
  optimizer.set_gradient_clipping(0);

  std::unordered_map<std::string, std::uint32_t> uint_configs {
    std::make_pair("Optimizer.epoch", 3),
  };
  std::unordered_map<std::string, float> float_configs {
    std::make_pair("MomentumSGD.eta", 1),
    std::make_pair("MomentumSGD.momentum", 2),
    std::make_pair("Optimizer.lr_scale", 4),
    std::make_pair("Optimizer.l2_strength", 5),
    std::make_pair("Optimizer.clip_threshold", 6),
  };
  optimizer.set_configs(uint_configs, float_configs);

  EXPECT_EQ(1, optimizer.eta());
  EXPECT_EQ(2, optimizer.momentum());
  EXPECT_EQ(3u, optimizer.get_epoch());
  EXPECT_EQ(4, optimizer.get_learning_rate_scaling());
  EXPECT_EQ(5, optimizer.get_weight_decay());
  EXPECT_EQ(6, optimizer.get_gradient_clipping());
}

TEST_F(OptimizerImplTest, CheckMomentumSGDUpdate) {
  Parameter param({2, 2}, {1, 2, 3, 4}, dev);
  ASSERT_TRUE(vector_match(
        vector<float> {1, 2, 3, 4}, param.value().to_vector()));

  MomentumSGD optimizer;
  optimizer.set_learning_rate_scaling(.1);
  optimizer.add(param);
  ASSERT_TRUE(param.has_stats("MomentumSGD.m"));
  EXPECT_TRUE(vector_match(
        vector<float>(4, 0), param.stats("MomentumSGD.m").to_vector()));

  const vector<vector<float>> expected_v {
    {9.9900000e-01, 1.9980000e+00, 2.9970000e+00, 3.9960000e+00},
    {9.9710100e-01, 1.9942020e+00, 2.9913030e+00, 3.9884040e+00},
    {9.9439480e-01, 1.9887896e+00, 2.9831844e+00, 3.9775792e+00},
    {9.9096482e-01, 1.9819296e+00, 2.9728945e+00, 3.9638593e+00},
    {9.8688688e-01, 1.9737738e+00, 2.9606606e+00, 3.9475475e+00},
  };
  const vector<vector<float>> expected_m {
    {-1.0000000e-03, -2.0000000e-03, -3.0000000e-03, -4.0000000e-03},
    {-1.8990000e-03, -3.7980000e-03, -5.6970000e-03, -7.5960000e-03},
    {-2.7062010e-03, -5.4124020e-03, -8.1186030e-03, -1.0824804e-02},
    {-3.4299757e-03, -6.8599514e-03, -1.0289927e-02, -1.3719903e-02},
    {-4.0779430e-03, -8.1558859e-03, -1.2233829e-02, -1.6311772e-02},
  };

  for (std::uint32_t i = 0; i < 5; ++i) {
    optimizer.reset_gradients();
    EXPECT_TRUE(vector_match(
          vector<float>(4, 0), param.gradient().to_vector()));

    param.gradient() += param.value();  // Squared loss
    optimizer.update();
    EXPECT_TRUE(vector_match(expected_v[i], param.value().to_vector()));
    EXPECT_TRUE(vector_match(
          expected_m[i], param.stats("MomentumSGD.m").to_vector()));
  }
}

TEST_F(OptimizerImplTest, CheckAdaGradSaveLoad) {
  AdaGrad optimizer(1, 2);
  optimizer.set_epoch(3);
  optimizer.set_learning_rate_scaling(4);
  optimizer.set_weight_decay(5);
  optimizer.set_gradient_clipping(6);

  const std::string path = "/tmp/primitiv_OptimizerImplTest_CheckAdaGradSaveLoad.data";
  optimizer.save(path);

  AdaGrad optimizer2;
  optimizer2.load(path);
  std::remove(path.c_str());

  EXPECT_EQ(1, optimizer2.eta());
  EXPECT_EQ(2, optimizer2.eps());
  EXPECT_EQ(3u, optimizer2.get_epoch());
  EXPECT_EQ(4, optimizer2.get_learning_rate_scaling());
  EXPECT_EQ(5, optimizer2.get_weight_decay());
  EXPECT_EQ(6, optimizer2.get_gradient_clipping());
}

TEST_F(OptimizerImplTest, CheckAdaGradGetConfigs) {
  AdaGrad optimizer(1, 2);
  optimizer.set_epoch(3);
  optimizer.set_learning_rate_scaling(4);
  optimizer.set_weight_decay(5);
  optimizer.set_gradient_clipping(6);

  std::unordered_map<std::string, std::uint32_t> uint_configs;
  std::unordered_map<std::string, float> float_configs;
  optimizer.get_configs(uint_configs, float_configs);

  EXPECT_EQ(1u, uint_configs.size());
  EXPECT_EQ(5u, float_configs.size());
  EXPECT_EQ(1, float_configs.at("AdaGrad.eta"));
  EXPECT_EQ(2, float_configs.at("AdaGrad.eps"));
  EXPECT_EQ(3u, uint_configs.at("Optimizer.epoch"));
  EXPECT_EQ(4, float_configs.at("Optimizer.lr_scale"));
  EXPECT_EQ(5, float_configs.at("Optimizer.l2_strength"));
  EXPECT_EQ(6, float_configs.at("Optimizer.clip_threshold"));
}

TEST_F(OptimizerImplTest, CheckAdaGradSetConfigs) {
  AdaGrad optimizer(0, 0);
  optimizer.set_epoch(0);
  optimizer.set_learning_rate_scaling(0);
  optimizer.set_weight_decay(0);
  optimizer.set_gradient_clipping(0);

  std::unordered_map<std::string, std::uint32_t> uint_configs {
    std::make_pair("Optimizer.epoch", 3),
  };
  std::unordered_map<std::string, float> float_configs {
    std::make_pair("AdaGrad.eta", 1),
    std::make_pair("AdaGrad.eps", 2),
    std::make_pair("Optimizer.lr_scale", 4),
    std::make_pair("Optimizer.l2_strength", 5),
    std::make_pair("Optimizer.clip_threshold", 6),
  };
  optimizer.set_configs(uint_configs, float_configs);

  EXPECT_EQ(1, optimizer.eta());
  EXPECT_EQ(2, optimizer.eps());
  EXPECT_EQ(3u, optimizer.get_epoch());
  EXPECT_EQ(4, optimizer.get_learning_rate_scaling());
  EXPECT_EQ(5, optimizer.get_weight_decay());
  EXPECT_EQ(6, optimizer.get_gradient_clipping());
}

TEST_F(OptimizerImplTest, CheckAdaGradUpdate) {
  Parameter param({2, 2}, {1, 2, 3, 4}, dev);
  ASSERT_TRUE(vector_match(
        vector<float> {1, 2, 3, 4}, param.value().to_vector()));

  AdaGrad optimizer;
  optimizer.set_learning_rate_scaling(.1);
  optimizer.add(param);
  ASSERT_TRUE(param.has_stats("AdaGrad.m"));
  EXPECT_TRUE(vector_match(
        vector<float>(4, 0), param.stats("AdaGrad.m").to_vector()));

  const vector<vector<float>> expected_v {
    {9.9989998e-01, 1.9999000e+00, 2.9999001e+00, 3.9999001e+00},
    {9.9982929e-01, 1.9998293e+00, 2.9998293e+00, 3.9998293e+00},
    {9.9977154e-01, 1.9997716e+00, 2.9997716e+00, 3.9997716e+00},
    {9.9972153e-01, 1.9997216e+00, 2.9997215e+00, 3.9997215e+00},
    {9.9967682e-01, 1.9996769e+00, 2.9996767e+00, 3.9996767e+00}
  };
  const vector<vector<float>> expected_m {
    {1.0000000e+00, 4.0000000e+00, 9.0000000e+00, 1.6000000e+01},
    {1.9998000e+00, 7.9995999e+00, 1.7999401e+01, 3.1999201e+01},
    {2.9994586e+00, 1.1998917e+01, 2.6998377e+01, 4.7997833e+01},
    {3.9990017e+00, 1.5998003e+01, 3.5997005e+01, 6.3996006e+01},
    {4.9984450e+00, 1.9996889e+01, 4.4995335e+01, 7.9993774e+01}
  };

  for (std::uint32_t i = 0; i < 5; ++i) {
    optimizer.reset_gradients();
    EXPECT_TRUE(vector_match(
          vector<float>(4, 0), param.gradient().to_vector()));

    param.gradient() += param.value();  // Squared loss
    optimizer.update();
    EXPECT_TRUE(vector_match(expected_v[i], param.value().to_vector()));
    EXPECT_TRUE(vector_match(
          expected_m[i], param.stats("AdaGrad.m").to_vector()));
  }
}

TEST_F(OptimizerImplTest, CheckRMSPropSaveLoad) {
  RMSProp optimizer(1, 2, 3);
  optimizer.set_epoch(4);
  optimizer.set_learning_rate_scaling(5);
  optimizer.set_weight_decay(6);
  optimizer.set_gradient_clipping(7);

  const std::string path = "/tmp/primitiv_OptimizerImplTest_CheckRMSPropSaveLoad.data";
  optimizer.save(path);

  RMSProp optimizer2;
  optimizer2.load(path);
  std::remove(path.c_str());

  EXPECT_EQ(1, optimizer2.eta());
  EXPECT_EQ(2, optimizer2.alpha());
  EXPECT_EQ(3, optimizer2.eps());
  EXPECT_EQ(4u, optimizer2.get_epoch());
  EXPECT_EQ(5, optimizer2.get_learning_rate_scaling());
  EXPECT_EQ(6, optimizer2.get_weight_decay());
  EXPECT_EQ(7, optimizer2.get_gradient_clipping());
}

TEST_F(OptimizerImplTest, CheckRMSPropGetConfigs) {
  RMSProp optimizer(1, 2, 3);
  optimizer.set_epoch(4);
  optimizer.set_learning_rate_scaling(5);
  optimizer.set_weight_decay(6);
  optimizer.set_gradient_clipping(7);

  std::unordered_map<std::string, std::uint32_t> uint_configs;
  std::unordered_map<std::string, float> float_configs;
  optimizer.get_configs(uint_configs, float_configs);

  EXPECT_EQ(1u, uint_configs.size());
  EXPECT_EQ(6u, float_configs.size());
  EXPECT_EQ(1, float_configs.at("RMSProp.eta"));
  EXPECT_EQ(2, float_configs.at("RMSProp.alpha"));
  EXPECT_EQ(3, float_configs.at("RMSProp.eps"));
  EXPECT_EQ(4u, uint_configs.at("Optimizer.epoch"));
  EXPECT_EQ(5, float_configs.at("Optimizer.lr_scale"));
  EXPECT_EQ(6, float_configs.at("Optimizer.l2_strength"));
  EXPECT_EQ(7, float_configs.at("Optimizer.clip_threshold"));
}

TEST_F(OptimizerImplTest, CheckRMSPropSetConfigs) {
  RMSProp optimizer(0, 0, 0);
  optimizer.set_epoch(0);
  optimizer.set_learning_rate_scaling(0);
  optimizer.set_weight_decay(0);
  optimizer.set_gradient_clipping(0);

  std::unordered_map<std::string, std::uint32_t> uint_configs {
    std::make_pair("Optimizer.epoch", 4),
  };
  std::unordered_map<std::string, float> float_configs {
    std::make_pair("RMSProp.eta", 1),
    std::make_pair("RMSProp.alpha", 2),
    std::make_pair("RMSProp.eps", 3),
    std::make_pair("Optimizer.lr_scale", 5),
    std::make_pair("Optimizer.l2_strength", 6),
    std::make_pair("Optimizer.clip_threshold", 7),
  };
  optimizer.set_configs(uint_configs, float_configs);

  EXPECT_EQ(1, optimizer.eta());
  EXPECT_EQ(2, optimizer.alpha());
  EXPECT_EQ(3, optimizer.eps());
  EXPECT_EQ(4u, optimizer.get_epoch());
  EXPECT_EQ(5, optimizer.get_learning_rate_scaling());
  EXPECT_EQ(6, optimizer.get_weight_decay());
  EXPECT_EQ(7, optimizer.get_gradient_clipping());
}

TEST_F(OptimizerImplTest, CheckRMSPropUpdate) {
  Parameter param({2, 2}, {1, 2, 3, 4}, dev);
  ASSERT_TRUE(vector_match(
        vector<float> {1, 2, 3, 4}, param.value().to_vector()));

  RMSProp optimizer;
  optimizer.set_learning_rate_scaling(.1);
  optimizer.add(param);
  ASSERT_TRUE(param.has_stats("RMSProp.m"));
  EXPECT_TRUE(vector_match(
        vector<float>(4, 0), param.stats("RMSProp.m").to_vector()));

  const vector<vector<float>> expected_v {
    {9.9683774e-01, 1.9968377e+00, 2.9968376e+00, 3.9968376e+00},
    {9.9454701e-01, 1.9945453e+00, 2.9945445e+00, 3.9945443e+00},
    {9.9263066e-01, 1.9926267e+00, 2.9926250e+00, 3.9926245e+00},
    {9.9093068e-01, 1.9909240e+00, 2.9909215e+00, 3.9909205e+00},
    {9.8937368e-01, 1.9893641e+00, 2.9893608e+00, 3.9893594e+00}
  };
  const vector<vector<float>> expected_m {
    {1.0000000e-01, 4.0000001e-01, 9.0000004e-01, 1.6000000e+00},
    {1.8936855e-01, 7.5873607e-01, 1.7081037e+00, 3.0374711e+00},
    {2.6934406e-01, 1.0806836e+00, 2.4340229e+00, 4.3293624e+00},
    {3.4094119e-01, 1.3696713e+00, 3.0862012e+00, 5.4905310e+00},
    {4.0504143e-01, 1.6290820e+00, 3.6721420e+00, 6.5342226e+00}
  };

  for (std::uint32_t i = 0; i < 5; ++i) {
    optimizer.reset_gradients();
    EXPECT_TRUE(vector_match(
          vector<float>(4, 0), param.gradient().to_vector()));

    param.gradient() += param.value();  // Squared loss
    optimizer.update();
    EXPECT_TRUE(vector_match(expected_v[i], param.value().to_vector()));
    EXPECT_TRUE(vector_match(
          expected_m[i], param.stats("RMSProp.m").to_vector()));
  }
}

TEST_F(OptimizerImplTest, CheckAdaDeltaSaveLoad) {
  AdaDelta optimizer(1, 2);
  optimizer.set_epoch(3);
  optimizer.set_learning_rate_scaling(4);
  optimizer.set_weight_decay(5);
  optimizer.set_gradient_clipping(6);

  const std::string path = "/tmp/primitiv_OptimizerImplTest_CheckAdaDeltaSaveLoad.data";
  optimizer.save(path);

  AdaDelta optimizer2;
  optimizer2.load(path);
  std::remove(path.c_str());

  EXPECT_EQ(1, optimizer2.rho());
  EXPECT_EQ(2, optimizer2.eps());
  EXPECT_EQ(3u, optimizer2.get_epoch());
  EXPECT_EQ(4, optimizer2.get_learning_rate_scaling());
  EXPECT_EQ(5, optimizer2.get_weight_decay());
  EXPECT_EQ(6, optimizer2.get_gradient_clipping());
}

TEST_F(OptimizerImplTest, CheckAdaDeltaGetConfigs) {
  AdaDelta optimizer(1, 2);
  optimizer.set_epoch(3);
  optimizer.set_learning_rate_scaling(4);
  optimizer.set_weight_decay(5);
  optimizer.set_gradient_clipping(6);

  std::unordered_map<std::string, std::uint32_t> uint_configs;
  std::unordered_map<std::string, float> float_configs;
  optimizer.get_configs(uint_configs, float_configs);

  EXPECT_EQ(1u, uint_configs.size());
  EXPECT_EQ(5u, float_configs.size());
  EXPECT_EQ(1, float_configs.at("AdaDelta.rho"));
  EXPECT_EQ(2, float_configs.at("AdaDelta.eps"));
  EXPECT_EQ(3u, uint_configs.at("Optimizer.epoch"));
  EXPECT_EQ(4, float_configs.at("Optimizer.lr_scale"));
  EXPECT_EQ(5, float_configs.at("Optimizer.l2_strength"));
  EXPECT_EQ(6, float_configs.at("Optimizer.clip_threshold"));
}

TEST_F(OptimizerImplTest, CheckAdaDeltaSetConfigs) {
  AdaDelta optimizer(0, 0);
  optimizer.set_epoch(0);
  optimizer.set_learning_rate_scaling(0);
  optimizer.set_weight_decay(0);
  optimizer.set_gradient_clipping(0);

  std::unordered_map<std::string, std::uint32_t> uint_configs {
    std::make_pair("Optimizer.epoch", 3),
  };
  std::unordered_map<std::string, float> float_configs {
    std::make_pair("AdaDelta.rho", 1),
    std::make_pair("AdaDelta.eps", 2),
    std::make_pair("Optimizer.lr_scale", 4),
    std::make_pair("Optimizer.l2_strength", 5),
    std::make_pair("Optimizer.clip_threshold", 6),
  };
  optimizer.set_configs(uint_configs, float_configs);

  EXPECT_EQ(1, optimizer.rho());
  EXPECT_EQ(2, optimizer.eps());
  EXPECT_EQ(3u, optimizer.get_epoch());
  EXPECT_EQ(4, optimizer.get_learning_rate_scaling());
  EXPECT_EQ(5, optimizer.get_weight_decay());
  EXPECT_EQ(6, optimizer.get_gradient_clipping());
}

TEST_F(OptimizerImplTest, CheckAdaDeltaUpdate) {
  Parameter param({2, 2}, {1, 2, 3, 4}, dev);
  ASSERT_TRUE(vector_match(
        vector<float> {1, 2, 3, 4}, param.value().to_vector()));

  AdaDelta optimizer;
  optimizer.set_learning_rate_scaling(.1);
  optimizer.add(param);
  ASSERT_TRUE(param.has_stats("AdaDelta.m1"));
  ASSERT_TRUE(param.has_stats("AdaDelta.m2"));
  EXPECT_TRUE(vector_match(
        vector<float>(4, 0), param.stats("AdaDelta.m1").to_vector()));
  EXPECT_TRUE(vector_match(
        vector<float>(4, 0), param.stats("AdaDelta.m2").to_vector()));

  const vector<vector<float>> expected_v {
    {9.9955279e-01, 1.9995528e+00, 2.9995528e+00, 3.9995528e+00},
    {9.9909998e-01, 1.9990999e+00, 2.9990999e+00, 3.9990999e+00},
    {9.9864346e-01, 1.9986433e+00, 2.9986432e+00, 3.9986432e+00},
    {9.9818414e-01, 1.9981838e+00, 2.9981836e+00, 3.9981836e+00},
    {9.9772260e-01, 1.9977219e+00, 2.9977217e+00, 3.9977216e+00},
  };
  const vector<vector<float>> expected_m1 {
    {9.9998000e-07, 9.9999500e-07, 9.9999778e-07, 9.9999875e-07},
    {1.9751542e-06, 1.9754076e-06, 1.9754876e-06, 1.9755268e-06},
    {2.9184791e-06, 2.9192686e-06, 2.9195252e-06, 2.9196523e-06},
    {3.8274016e-06, 3.8290573e-06, 3.8296005e-06, 3.8298705e-06},
    {4.7011537e-06, 4.7040199e-06, 4.7049647e-06, 4.7054352e-06},
  };
  const vector<vector<float>> expected_m2 {
    {5.0000000e-02, 2.0000000e-01, 4.5000000e-01, 8.0000000e-01},
    {9.7455289e-02, 3.8991057e-01, 8.7736585e-01, 1.5598211e+00},
    {1.4249256e-01, 5.7023507e-01, 1.2832276e+00, 2.2814701e+00},
    {1.8523237e-01, 7.4145206e-01, 1.6686592e+00, 2.9668539e+00},
    {2.2578933e-01, 9.0401638e-01, 2.0346815e+00, 3.6177848e+00},
  };

  for (std::uint32_t i = 0; i < 5; ++i) {
    optimizer.reset_gradients();
    EXPECT_TRUE(vector_match(
          vector<float>(4, 0), param.gradient().to_vector()));

    param.gradient() += param.value();  // Squared loss
    optimizer.update();
    EXPECT_TRUE(vector_match(expected_v[i], param.value().to_vector()));
    EXPECT_TRUE(vector_match(
          expected_m1[i], param.stats("AdaDelta.m1").to_vector()));
    EXPECT_TRUE(vector_match(
          expected_m2[i], param.stats("AdaDelta.m2").to_vector()));
  }
}

TEST_F(OptimizerImplTest, CheckAdamSaveLoad) {
  Adam optimizer(1, 2, 3, 4);
  optimizer.set_epoch(5);
  optimizer.set_learning_rate_scaling(6);
  optimizer.set_weight_decay(7);
  optimizer.set_gradient_clipping(8);

  const std::string path = "/tmp/primitiv_OptimizerImplTest_CheckAdamSaveLoad.data";
  optimizer.save(path);

  Adam optimizer2;
  optimizer2.load(path);
  std::remove(path.c_str());

  EXPECT_EQ(1, optimizer2.alpha());
  EXPECT_EQ(2, optimizer2.beta1());
  EXPECT_EQ(3, optimizer2.beta2());
  EXPECT_EQ(4, optimizer2.eps());
  EXPECT_EQ(5u, optimizer2.get_epoch());
  EXPECT_EQ(6, optimizer2.get_learning_rate_scaling());
  EXPECT_EQ(7, optimizer2.get_weight_decay());
  EXPECT_EQ(8, optimizer2.get_gradient_clipping());
}

TEST_F(OptimizerImplTest, CheckAdamGetConfigs) {
  Adam optimizer(1, 2, 3, 4);
  optimizer.set_epoch(5);
  optimizer.set_learning_rate_scaling(6);
  optimizer.set_weight_decay(7);
  optimizer.set_gradient_clipping(8);

  std::unordered_map<std::string, std::uint32_t> uint_configs;
  std::unordered_map<std::string, float> float_configs;
  optimizer.get_configs(uint_configs, float_configs);

  EXPECT_EQ(1u, uint_configs.size());
  EXPECT_EQ(7u, float_configs.size());
  EXPECT_EQ(1, float_configs.at("Adam.alpha"));
  EXPECT_EQ(2, float_configs.at("Adam.beta1"));
  EXPECT_EQ(3, float_configs.at("Adam.beta2"));
  EXPECT_EQ(4, float_configs.at("Adam.eps"));
  EXPECT_EQ(5u, uint_configs.at("Optimizer.epoch"));
  EXPECT_EQ(6, float_configs.at("Optimizer.lr_scale"));
  EXPECT_EQ(7, float_configs.at("Optimizer.l2_strength"));
  EXPECT_EQ(8, float_configs.at("Optimizer.clip_threshold"));
}

TEST_F(OptimizerImplTest, CheckAdamSetConfigs) {
  Adam optimizer(0, 0, 0, 0);
  optimizer.set_epoch(0);
  optimizer.set_learning_rate_scaling(0);
  optimizer.set_weight_decay(0);
  optimizer.set_gradient_clipping(0);

  std::unordered_map<std::string, std::uint32_t> uint_configs {
    std::make_pair("Optimizer.epoch", 5),
  };
  std::unordered_map<std::string, float> float_configs {
    std::make_pair("Adam.alpha", 1),
    std::make_pair("Adam.beta1", 2),
    std::make_pair("Adam.beta2", 3),
    std::make_pair("Adam.eps", 4),
    std::make_pair("Optimizer.lr_scale", 6),
    std::make_pair("Optimizer.l2_strength", 7),
    std::make_pair("Optimizer.clip_threshold", 8),
  };
  optimizer.set_configs(uint_configs, float_configs);

  EXPECT_EQ(1, optimizer.alpha());
  EXPECT_EQ(2, optimizer.beta1());
  EXPECT_EQ(3, optimizer.beta2());
  EXPECT_EQ(4, optimizer.eps());
  EXPECT_EQ(5u, optimizer.get_epoch());
  EXPECT_EQ(6, optimizer.get_learning_rate_scaling());
  EXPECT_EQ(7, optimizer.get_weight_decay());
  EXPECT_EQ(8, optimizer.get_gradient_clipping());
}

TEST_F(OptimizerImplTest, CheckAdamUpdate) {
  Parameter param({2, 2}, {1, 2, 3, 4}, dev);
  ASSERT_TRUE(vector_match(
        vector<float> {1, 2, 3, 4}, param.value().to_vector()));

  Adam optimizer;
  optimizer.set_learning_rate_scaling(.1);
  optimizer.add(param);
  ASSERT_TRUE(param.has_stats("Adam.m1"));
  ASSERT_TRUE(param.has_stats("Adam.m2"));
  EXPECT_TRUE(vector_match(
        vector<float>(4, 0), param.stats("Adam.m1").to_vector()));
  EXPECT_TRUE(vector_match(
        vector<float>(4, 0), param.stats("Adam.m2").to_vector()));

  const vector<vector<float>> expected_v {
    {9.9990000e-01, 1.9999000e+00, 2.9999000e+00, 3.9999000e+00},
    {9.9980000e-01, 1.9998000e+00, 2.9998000e+00, 3.9998000e+00},
    {9.9970000e-01, 1.9997000e+00, 2.9997000e+00, 3.9997000e+00},
    {9.9960000e-01, 1.9996000e+00, 2.9996000e+00, 3.9996000e+00},
    {9.9950000e-01, 1.9995000e+00, 2.9995000e+00, 3.9995000e+00},
  };
  const vector<vector<float>> expected_m1 {
    {1.0000000e-01, 2.0000000e-01, 3.0000000e-01, 4.0000000e-01},
    {1.8999000e-01, 3.7999000e-01, 5.6999000e-01, 7.5999000e-01},
    {2.7097100e-01, 5.4197100e-01, 8.1297100e-01, 1.0839710e+00},
    {3.4384390e-01, 6.8774390e-01, 1.0316439e+00, 1.3755439e+00},
    {4.0941951e-01, 8.1892951e-01, 1.2284395e+00, 1.6379495e+00},
  };
  const vector<vector<float>> expected_m2 {
    {1.0000000e-03, 4.0000000e-03, 9.0000000e-03, 1.6000000e-02},
    {1.9988000e-03, 7.9956000e-03, 1.7990400e-02, 3.1983200e-02},
    {2.9964013e-03, 1.1986804e-02, 2.6971210e-02, 4.7949617e-02},
    {3.9928049e-03, 1.5973618e-02, 3.5942439e-02, 6.3899267e-02},
    {4.9880123e-03, 1.9956044e-02, 4.4904096e-02, 7.9832168e-02},
  };

  for (std::uint32_t i = 0; i < 5; ++i) {
    optimizer.reset_gradients();
    EXPECT_TRUE(vector_match(
          vector<float>(4, 0), param.gradient().to_vector()));

    param.gradient() += param.value();  // Squared loss
    optimizer.update();
    EXPECT_TRUE(vector_near(
          expected_v[i], param.value().to_vector(), 1e-5));
    EXPECT_TRUE(vector_near(
          expected_m1[i], param.stats("Adam.m1").to_vector(), 1e-5));
    EXPECT_TRUE(vector_near(
          expected_m2[i], param.stats("Adam.m2").to_vector(), 1e-5));
  }
}

}  // namespace optimizers
}  // namespace primitiv
