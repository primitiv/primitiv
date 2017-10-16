#include <config.h>

#include <cstdio>
#include <gtest/gtest.h>
#include <primitiv/error.h>
#include <primitiv/naive_device.h>
#include <primitiv/parameter.h>
#include <primitiv/trainer_impl.h>
#include <test_utils.h>

using std::vector;
using test_utils::vector_match;
using test_utils::vector_near;

namespace primitiv {
namespace trainers {

class TrainerImplTest : public testing::Test {
protected:
  devices::Naive dev;
};

TEST_F(TrainerImplTest, CheckNames) {
  SGD sgd;
  EXPECT_EQ("SGD", sgd.name());
  MomentumSGD momentumsgd;
  EXPECT_EQ("MomentumSGD", momentumsgd.name());
  AdaGrad adagrad;
  EXPECT_EQ("AdaGrad", adagrad.name());
  Adam adam;
  EXPECT_EQ("Adam", adam.name());
}

TEST_F(TrainerImplTest, CheckDefaultHyperparameters) {
  SGD sgd;
  EXPECT_FLOAT_EQ(.1, sgd.eta());

  MomentumSGD momentumsgd;
  EXPECT_FLOAT_EQ(.01, momentumsgd.eta());
  EXPECT_FLOAT_EQ(.9, momentumsgd.momentum());

  AdaGrad adagrad;
  EXPECT_FLOAT_EQ(.001, adagrad.eta());
  EXPECT_FLOAT_EQ(1e-8, adagrad.eps());

  Adam adam;
  EXPECT_FLOAT_EQ(.001, adam.alpha());
  EXPECT_FLOAT_EQ(.9, adam.beta1());
  EXPECT_FLOAT_EQ(.999, adam.beta2());
  EXPECT_FLOAT_EQ(1e-8, adam.eps());
}

TEST_F(TrainerImplTest, CheckGivenHyperparameters) {
  SGD sgd(1);
  EXPECT_FLOAT_EQ(1, sgd.eta());

  MomentumSGD momentumsgd(1, 2);
  EXPECT_FLOAT_EQ(1, momentumsgd.eta());
  EXPECT_FLOAT_EQ(2, momentumsgd.momentum());

  AdaGrad adagrad(1, 2);
  EXPECT_FLOAT_EQ(1, adagrad.eta());
  EXPECT_FLOAT_EQ(2, adagrad.eps());

  Adam adam(1, 2, 3, 4);
  EXPECT_FLOAT_EQ(1, adam.alpha());
  EXPECT_FLOAT_EQ(2, adam.beta1());
  EXPECT_FLOAT_EQ(3, adam.beta2());
  EXPECT_FLOAT_EQ(4, adam.eps());
}

TEST_F(TrainerImplTest, CheckInvalidSetConfigsByFile) {
  SGD sgd;

  const std::string path = "/tmp/primitiv_TrainerImplTest_CheckInvalidSetConfigsByFile.data";
  sgd.save(path);

  MomentumSGD momentumsgd;
  EXPECT_THROW(momentumsgd.set_configs_by_file(path), Error);

  AdaGrad adagrad;
  EXPECT_THROW(adagrad.set_configs_by_file(path), Error);

  Adam adam;
  EXPECT_THROW(adam.set_configs_by_file(path), Error);

  std::remove(path.c_str());
}

TEST_F(TrainerImplTest, CheckSGDSaveLoad) {
  SGD sgd(1);
  sgd.set_epoch(2);
  sgd.set_learning_rate_scaling(3);
  sgd.set_weight_decay(4);
  sgd.set_gradient_clipping(5);

  const std::string path = "/tmp/primitiv_TrainerImplTest_CheckSGDSaveLoad.data";
  sgd.save(path);

  EXPECT_EQ("SGD", Trainer::detect_name(path));

  std::shared_ptr<Trainer> loaded = Trainer::load(path);
  std::remove(path.c_str());

  ASSERT_EQ("SGD", loaded->name());

  std::shared_ptr<SGD> sgd2 = std::static_pointer_cast<SGD>(loaded);
  EXPECT_EQ(1, sgd2->eta());
  EXPECT_EQ(2, sgd2->get_epoch());
  EXPECT_EQ(3, sgd2->get_learning_rate_scaling());
  EXPECT_EQ(4, sgd2->get_weight_decay());
  EXPECT_EQ(5, sgd2->get_gradient_clipping());
}

TEST_F(TrainerImplTest, CheckSGDGetConfigs) {
  SGD sgd(1);
  sgd.set_epoch(2);
  sgd.set_learning_rate_scaling(3);
  sgd.set_weight_decay(4);
  sgd.set_gradient_clipping(5);

  std::unordered_map<std::string, unsigned> uint_configs;
  std::unordered_map<std::string, float> float_configs;
  sgd.get_configs(uint_configs, float_configs);

  EXPECT_EQ(1u, uint_configs.size());
  EXPECT_EQ(4u, float_configs.size());
  EXPECT_EQ(1, float_configs.at("SGD.eta"));
  EXPECT_EQ(2, uint_configs.at("Trainer.epoch"));
  EXPECT_EQ(3, float_configs.at("Trainer.lr_scale"));
  EXPECT_EQ(4, float_configs.at("Trainer.l2_strength"));
  EXPECT_EQ(5, float_configs.at("Trainer.clip_threshold"));
}

TEST_F(TrainerImplTest, CheckSGDSetConfigs) {
  SGD sgd(0);
  sgd.set_epoch(0);
  sgd.set_learning_rate_scaling(0);
  sgd.set_weight_decay(0);
  sgd.set_gradient_clipping(0);

  std::unordered_map<std::string, unsigned> uint_configs {
    std::make_pair("Trainer.epoch", 2),
  };
  std::unordered_map<std::string, float> float_configs {
    std::make_pair("SGD.eta", 1),
    std::make_pair("Trainer.lr_scale", 3),
    std::make_pair("Trainer.l2_strength", 4),
    std::make_pair("Trainer.clip_threshold", 5),
  };
  sgd.set_configs(uint_configs, float_configs);

  EXPECT_EQ(1, sgd.eta());
  EXPECT_EQ(2, sgd.get_epoch());
  EXPECT_EQ(3, sgd.get_learning_rate_scaling());
  EXPECT_EQ(4, sgd.get_weight_decay());
  EXPECT_EQ(5, sgd.get_gradient_clipping());
}

TEST_F(TrainerImplTest, CheckSGDSetConfigsByFile) {
  SGD sgd(1);
  sgd.set_epoch(2);
  sgd.set_learning_rate_scaling(3);
  sgd.set_weight_decay(4);
  sgd.set_gradient_clipping(5);

  const std::string path = "/tmp/primitiv_TrainerImplTest_CheckSGDSetConfigsFromFile.data";
  sgd.save(path);

  SGD sgd2(0);
  sgd2.set_epoch(0);
  sgd2.set_learning_rate_scaling(0);
  sgd2.set_weight_decay(0);
  sgd2.set_gradient_clipping(0);

  sgd2.set_configs_by_file(path);
  std::remove(path.c_str());

  EXPECT_EQ(1, sgd2.eta());
  EXPECT_EQ(2, sgd2.get_epoch());
  EXPECT_EQ(3, sgd2.get_learning_rate_scaling());
  EXPECT_EQ(4, sgd2.get_weight_decay());
  EXPECT_EQ(5, sgd2.get_gradient_clipping());
}

TEST_F(TrainerImplTest, CheckSGDUpdate) {
  Parameter param("param", {2, 2}, {1, 2, 3, 4}, dev);
  ASSERT_TRUE(vector_match(
        vector<float> {1, 2, 3, 4}, param.value().to_vector()));

  SGD trainer;
  trainer.set_learning_rate_scaling(.1);
  trainer.add_parameter(param);

  vector<vector<float>> expected_v {
    {9.9000000e-01, 1.9800000e+00, 2.9700000e+00, 3.9600000e+00},
    {9.8010000e-01, 1.9602000e+00, 2.9403000e+00, 3.9204000e+00},
    {9.7029900e-01, 1.9405980e+00, 2.9108970e+00, 3.8811960e+00},
    {9.6059601e-01, 1.9211920e+00, 2.8817880e+00, 3.8423840e+00},
    {9.5099005e-01, 1.9019801e+00, 2.8529701e+00, 3.8039602e+00},
  };

  for (unsigned i = 0; i < 5; ++i) {
    trainer.reset_gradients();
    EXPECT_TRUE(vector_match(
          vector<float>(4, 0), param.gradient().to_vector()));

    param.gradient() += param.value();  // Squared loss
    trainer.update();
    EXPECT_TRUE(vector_match(expected_v[i], param.value().to_vector()));
  }
}

TEST_F(TrainerImplTest, CheckMomentumSGDSaveLoad) {
  MomentumSGD trainer(1, 2);
  trainer.set_epoch(3);
  trainer.set_learning_rate_scaling(4);
  trainer.set_weight_decay(5);
  trainer.set_gradient_clipping(6);

  const std::string path = "/tmp/primitiv_TrainerImplTest_CheckMomentumSGDSaveLoad.data";
  trainer.save(path);

  EXPECT_EQ("MomentumSGD", Trainer::detect_name(path));

  std::shared_ptr<Trainer> loaded = Trainer::load(path);
  std::remove(path.c_str());

  ASSERT_EQ("MomentumSGD", loaded->name());

  std::shared_ptr<MomentumSGD> trainer2 = std::static_pointer_cast<MomentumSGD>(loaded);
  EXPECT_EQ(1, trainer2->eta());
  EXPECT_EQ(2, trainer2->momentum());
  EXPECT_EQ(3, trainer2->get_epoch());
  EXPECT_EQ(4, trainer2->get_learning_rate_scaling());
  EXPECT_EQ(5, trainer2->get_weight_decay());
  EXPECT_EQ(6, trainer2->get_gradient_clipping());
}

TEST_F(TrainerImplTest, CheckMomentumSGDGetConfigs) {
  MomentumSGD trainer(1, 2);
  trainer.set_epoch(3);
  trainer.set_learning_rate_scaling(4);
  trainer.set_weight_decay(5);
  trainer.set_gradient_clipping(6);

  std::unordered_map<std::string, unsigned> uint_configs;
  std::unordered_map<std::string, float> float_configs;
  trainer.get_configs(uint_configs, float_configs);

  EXPECT_EQ(1u, uint_configs.size());
  EXPECT_EQ(5u, float_configs.size());
  EXPECT_EQ(1, float_configs.at("MomentumSGD.eta"));
  EXPECT_EQ(2, float_configs.at("MomentumSGD.momentum"));
  EXPECT_EQ(3, uint_configs.at("Trainer.epoch"));
  EXPECT_EQ(4, float_configs.at("Trainer.lr_scale"));
  EXPECT_EQ(5, float_configs.at("Trainer.l2_strength"));
  EXPECT_EQ(6, float_configs.at("Trainer.clip_threshold"));
}

TEST_F(TrainerImplTest, CheckMomentumSGDSetConfigs) {
  MomentumSGD trainer(0, 0);
  trainer.set_epoch(0);
  trainer.set_learning_rate_scaling(0);
  trainer.set_weight_decay(0);
  trainer.set_gradient_clipping(0);

  std::unordered_map<std::string, unsigned> uint_configs {
    std::make_pair("Trainer.epoch", 3),
  };
  std::unordered_map<std::string, float> float_configs {
    std::make_pair("MomentumSGD.eta", 1),
    std::make_pair("MomentumSGD.momentum", 2),
    std::make_pair("Trainer.lr_scale", 4),
    std::make_pair("Trainer.l2_strength", 5),
    std::make_pair("Trainer.clip_threshold", 6),
  };
  trainer.set_configs(uint_configs, float_configs);

  EXPECT_EQ(1, trainer.eta());
  EXPECT_EQ(2, trainer.momentum());
  EXPECT_EQ(3, trainer.get_epoch());
  EXPECT_EQ(4, trainer.get_learning_rate_scaling());
  EXPECT_EQ(5, trainer.get_weight_decay());
  EXPECT_EQ(6, trainer.get_gradient_clipping());
}

TEST_F(TrainerImplTest, CheckMomentumSGDSetConfigsByFile) {
  MomentumSGD trainer(1, 2);
  trainer.set_epoch(3);
  trainer.set_learning_rate_scaling(4);
  trainer.set_weight_decay(5);
  trainer.set_gradient_clipping(6);

  const std::string path = "/tmp/primitiv_TrainerImplTest_CheckMomentumSGDSetConfigsFromFile.data";
  trainer.save(path);

  MomentumSGD trainer2(0, 0);
  trainer.set_epoch(0);
  trainer.set_learning_rate_scaling(0);
  trainer.set_weight_decay(0);
  trainer.set_gradient_clipping(0);

  trainer2.set_configs_by_file(path);
  std::remove(path.c_str());

  EXPECT_EQ(1, trainer2.eta());
  EXPECT_EQ(2, trainer2.momentum());
  EXPECT_EQ(3, trainer2.get_epoch());
  EXPECT_EQ(4, trainer2.get_learning_rate_scaling());
  EXPECT_EQ(5, trainer2.get_weight_decay());
  EXPECT_EQ(6, trainer2.get_gradient_clipping());
}

TEST_F(TrainerImplTest, CheckMomentumSGDUpdate) {
  Parameter param("param", {2, 2}, {1, 2, 3, 4}, dev);
  ASSERT_TRUE(vector_match(
        vector<float> {1, 2, 3, 4}, param.value().to_vector()));

  MomentumSGD trainer;
  trainer.set_learning_rate_scaling(.1);
  trainer.add_parameter(param);
  ASSERT_TRUE(vector_match(
        vector<float>(4, 0), param.stats("momentumsgd-m").to_vector()));

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

  for (unsigned i = 0; i < 5; ++i) {
    trainer.reset_gradients();
    EXPECT_TRUE(vector_match(
          vector<float>(4, 0), param.gradient().to_vector()));

    param.gradient() += param.value();  // Squared loss
    trainer.update();
    EXPECT_TRUE(vector_match(expected_v[i], param.value().to_vector()));
    EXPECT_TRUE(vector_match(
          expected_m[i], param.stats("momentumsgd-m").to_vector()));
  }
}

TEST_F(TrainerImplTest, CheckAdaGradSaveLoad) {
  AdaGrad adagrad(1, 2);
  adagrad.set_epoch(3);
  adagrad.set_learning_rate_scaling(4);
  adagrad.set_weight_decay(5);
  adagrad.set_gradient_clipping(6);

  const std::string path = "/tmp/primitiv_TrainerImplTest_CheckAdaGradSaveLoad.data";
  adagrad.save(path);

  EXPECT_EQ("AdaGrad", Trainer::detect_name(path));

  std::shared_ptr<Trainer> loaded = Trainer::load(path);
  std::remove(path.c_str());

  ASSERT_EQ("AdaGrad", loaded->name());

  std::shared_ptr<AdaGrad> adagrad2 = std::static_pointer_cast<AdaGrad>(loaded);
  EXPECT_EQ(1, adagrad2->eta());
  EXPECT_EQ(2, adagrad2->eps());
  EXPECT_EQ(3, adagrad2->get_epoch());
  EXPECT_EQ(4, adagrad2->get_learning_rate_scaling());
  EXPECT_EQ(5, adagrad2->get_weight_decay());
  EXPECT_EQ(6, adagrad2->get_gradient_clipping());
}

TEST_F(TrainerImplTest, CheckAdaGradGetConfigs) {
  AdaGrad adagrad(1, 2);
  adagrad.set_epoch(3);
  adagrad.set_learning_rate_scaling(4);
  adagrad.set_weight_decay(5);
  adagrad.set_gradient_clipping(6);

  std::unordered_map<std::string, unsigned> uint_configs;
  std::unordered_map<std::string, float> float_configs;
  adagrad.get_configs(uint_configs, float_configs);

  EXPECT_EQ(1u, uint_configs.size());
  EXPECT_EQ(5u, float_configs.size());
  EXPECT_EQ(1, float_configs.at("AdaGrad.eta"));
  EXPECT_EQ(2, float_configs.at("AdaGrad.eps"));
  EXPECT_EQ(3, uint_configs.at("Trainer.epoch"));
  EXPECT_EQ(4, float_configs.at("Trainer.lr_scale"));
  EXPECT_EQ(5, float_configs.at("Trainer.l2_strength"));
  EXPECT_EQ(6, float_configs.at("Trainer.clip_threshold"));
}

TEST_F(TrainerImplTest, CheckAdaGradSetConfigs) {
  AdaGrad adagrad(0, 0);
  adagrad.set_epoch(0);
  adagrad.set_learning_rate_scaling(0);
  adagrad.set_weight_decay(0);
  adagrad.set_gradient_clipping(0);

  std::unordered_map<std::string, unsigned> uint_configs {
    std::make_pair("Trainer.epoch", 3),
  };
  std::unordered_map<std::string, float> float_configs {
    std::make_pair("AdaGrad.eta", 1),
    std::make_pair("AdaGrad.eps", 2),
    std::make_pair("Trainer.lr_scale", 4),
    std::make_pair("Trainer.l2_strength", 5),
    std::make_pair("Trainer.clip_threshold", 6),
  };
  adagrad.set_configs(uint_configs, float_configs);

  EXPECT_EQ(1, adagrad.eta());
  EXPECT_EQ(2, adagrad.eps());
  EXPECT_EQ(3, adagrad.get_epoch());
  EXPECT_EQ(4, adagrad.get_learning_rate_scaling());
  EXPECT_EQ(5, adagrad.get_weight_decay());
  EXPECT_EQ(6, adagrad.get_gradient_clipping());
}

TEST_F(TrainerImplTest, CheckAdaGradSetConfigsByFile) {
  AdaGrad adagrad(1, 2);
  adagrad.set_epoch(3);
  adagrad.set_learning_rate_scaling(4);
  adagrad.set_weight_decay(5);
  adagrad.set_gradient_clipping(6);

  const std::string path = "/tmp/primitiv_TrainerImplTest_CheckAdaGradSetConfigsFromFile.data";
  adagrad.save(path);

  AdaGrad adagrad2(0, 0);
  adagrad2.set_epoch(0);
  adagrad2.set_learning_rate_scaling(0);
  adagrad2.set_weight_decay(0);
  adagrad2.set_gradient_clipping(0);

  adagrad2.set_configs_by_file(path);
  std::remove(path.c_str());

  EXPECT_EQ(1, adagrad2.eta());
  EXPECT_EQ(2, adagrad2.eps());
  EXPECT_EQ(3, adagrad2.get_epoch());
  EXPECT_EQ(4, adagrad2.get_learning_rate_scaling());
  EXPECT_EQ(5, adagrad2.get_weight_decay());
  EXPECT_EQ(6, adagrad2.get_gradient_clipping());
}

TEST_F(TrainerImplTest, CheckAdaGradUpdate) {
  Parameter param("param", {2, 2}, {1, 2, 3, 4}, dev);
  ASSERT_TRUE(vector_match(
        vector<float> {1, 2, 3, 4}, param.value().to_vector()));

  AdaGrad trainer;
  trainer.set_learning_rate_scaling(.1);
  trainer.add_parameter(param);
  ASSERT_TRUE(param.has_stats("adagrad-m"));
  EXPECT_TRUE(vector_match(
        vector<float>(4, 0), param.stats("adagrad-m").to_vector()));

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

  for (unsigned i = 0; i < 5; ++i) {
    trainer.reset_gradients();
    EXPECT_TRUE(vector_match(
          vector<float>(4, 0), param.gradient().to_vector()));

    param.gradient() += param.value();  // Squared loss
    trainer.update();
    EXPECT_TRUE(vector_near(
          expected_v[i], param.value().to_vector(), 1e-5));
    EXPECT_TRUE(vector_near(
          expected_m[i], param.stats("adagrad-m").to_vector(), 1e-5));
  }
}

TEST_F(TrainerImplTest, CheckAdamSaveLoad) {
  Adam adam(1, 2, 3, 4);
  adam.set_epoch(5);
  adam.set_learning_rate_scaling(6);
  adam.set_weight_decay(7);
  adam.set_gradient_clipping(8);

  const std::string path = "/tmp/primitiv_TrainerImplTest_CheckAdamSaveLoad.data";
  adam.save(path);

  EXPECT_EQ("Adam", Trainer::detect_name(path));

  std::shared_ptr<Trainer> loaded = Trainer::load(path);
  std::remove(path.c_str());

  ASSERT_EQ("Adam", loaded->name());

  std::shared_ptr<Adam> adam2 = std::static_pointer_cast<Adam>(loaded);
  EXPECT_EQ(1, adam2->alpha());
  EXPECT_EQ(2, adam2->beta1());
  EXPECT_EQ(3, adam2->beta2());
  EXPECT_EQ(4, adam2->eps());
  EXPECT_EQ(5, adam2->get_epoch());
  EXPECT_EQ(6, adam2->get_learning_rate_scaling());
  EXPECT_EQ(7, adam2->get_weight_decay());
  EXPECT_EQ(8, adam2->get_gradient_clipping());
}

TEST_F(TrainerImplTest, CheckAdamGetConfigs) {
  Adam adam(1, 2, 3, 4);
  adam.set_epoch(5);
  adam.set_learning_rate_scaling(6);
  adam.set_weight_decay(7);
  adam.set_gradient_clipping(8);

  std::unordered_map<std::string, unsigned> uint_configs;
  std::unordered_map<std::string, float> float_configs;
  adam.get_configs(uint_configs, float_configs);

  EXPECT_EQ(1u, uint_configs.size());
  EXPECT_EQ(7u, float_configs.size());
  EXPECT_EQ(1, float_configs.at("Adam.alpha"));
  EXPECT_EQ(2, float_configs.at("Adam.beta1"));
  EXPECT_EQ(3, float_configs.at("Adam.beta2"));
  EXPECT_EQ(4, float_configs.at("Adam.eps"));
  EXPECT_EQ(5, uint_configs.at("Trainer.epoch"));
  EXPECT_EQ(6, float_configs.at("Trainer.lr_scale"));
  EXPECT_EQ(7, float_configs.at("Trainer.l2_strength"));
  EXPECT_EQ(8, float_configs.at("Trainer.clip_threshold"));
}

TEST_F(TrainerImplTest, CheckAdamSetConfigs) {
  Adam adam(0, 0, 0, 0);
  adam.set_epoch(0);
  adam.set_learning_rate_scaling(0);
  adam.set_weight_decay(0);
  adam.set_gradient_clipping(0);

  std::unordered_map<std::string, unsigned> uint_configs {
    std::make_pair("Trainer.epoch", 5),
  };
  std::unordered_map<std::string, float> float_configs {
    std::make_pair("Adam.alpha", 1),
    std::make_pair("Adam.beta1", 2),
    std::make_pair("Adam.beta2", 3),
    std::make_pair("Adam.eps", 4),
    std::make_pair("Trainer.lr_scale", 6),
    std::make_pair("Trainer.l2_strength", 7),
    std::make_pair("Trainer.clip_threshold", 8),
  };
  adam.set_configs(uint_configs, float_configs);

  EXPECT_EQ(1, adam.alpha());
  EXPECT_EQ(2, adam.beta1());
  EXPECT_EQ(3, adam.beta2());
  EXPECT_EQ(4, adam.eps());
  EXPECT_EQ(5, adam.get_epoch());
  EXPECT_EQ(6, adam.get_learning_rate_scaling());
  EXPECT_EQ(7, adam.get_weight_decay());
  EXPECT_EQ(8, adam.get_gradient_clipping());
}

TEST_F(TrainerImplTest, CheckAdamSetConfigsByFile) {
  Adam adam(1, 2, 3, 4);
  adam.set_epoch(5);
  adam.set_learning_rate_scaling(6);
  adam.set_weight_decay(7);
  adam.set_gradient_clipping(8);

  const std::string path = "/tmp/primitiv_TrainerImplTest_CheckAdamSetConfigsFromFile.data";
  adam.save(path);

  Adam adam2(0, 0, 0, 0);
  adam2.set_epoch(0);
  adam2.set_learning_rate_scaling(0);
  adam2.set_weight_decay(0);
  adam2.set_gradient_clipping(0);

  adam2.set_configs_by_file(path);
  std::remove(path.c_str());

  EXPECT_EQ(1, adam2.alpha());
  EXPECT_EQ(2, adam2.beta1());
  EXPECT_EQ(3, adam2.beta2());
  EXPECT_EQ(4, adam2.eps());
  EXPECT_EQ(5, adam2.get_epoch());
  EXPECT_EQ(6, adam2.get_learning_rate_scaling());
  EXPECT_EQ(7, adam2.get_weight_decay());
  EXPECT_EQ(8, adam2.get_gradient_clipping());
}

TEST_F(TrainerImplTest, CheckAdamUpdate) {
  Parameter param("param", {2, 2}, {1, 2, 3, 4}, dev);
  ASSERT_TRUE(vector_match(
        vector<float> {1, 2, 3, 4}, param.value().to_vector()));

  Adam trainer;
  trainer.set_learning_rate_scaling(.1);
  trainer.add_parameter(param);
  ASSERT_TRUE(param.has_stats("adam-m1"));
  ASSERT_TRUE(param.has_stats("adam-m2"));
  EXPECT_TRUE(vector_match(
        vector<float>(4, 0), param.stats("adam-m1").to_vector()));
  EXPECT_TRUE(vector_match(
        vector<float>(4, 0), param.stats("adam-m2").to_vector()));

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

  for (unsigned i = 0; i < 5; ++i) {
    trainer.reset_gradients();
    EXPECT_TRUE(vector_match(
          vector<float>(4, 0), param.gradient().to_vector()));

    param.gradient() += param.value();  // Squared loss
    trainer.update();
    EXPECT_TRUE(vector_near(
          expected_v[i], param.value().to_vector(), 1e-5));
    EXPECT_TRUE(vector_near(
          expected_m1[i], param.stats("adam-m1").to_vector(), 1e-5));
    EXPECT_TRUE(vector_near(
          expected_m2[i], param.stats("adam-m2").to_vector(), 1e-5));
  }
}

}  // namespace trainers
}  // namespace primitiv
