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
  Adam adam;
  EXPECT_EQ("Adam", adam.name());
}

TEST_F(TrainerImplTest, CheckDefaultHyperparameters) {
  SGD sgd;
  EXPECT_FLOAT_EQ(.1, sgd.eta());

  Adam adam;
  EXPECT_FLOAT_EQ(.001, adam.alpha());
  EXPECT_FLOAT_EQ(.9, adam.beta1());
  EXPECT_FLOAT_EQ(.999, adam.beta2());
  EXPECT_FLOAT_EQ(1e-8, adam.eps());
}

TEST_F(TrainerImplTest, CheckGivenHyperparameters) {
  SGD sgd(1);
  EXPECT_FLOAT_EQ(1, sgd.eta());

  Adam adam(1, 2, 3, 4);
  EXPECT_FLOAT_EQ(1, adam.alpha());
  EXPECT_FLOAT_EQ(2, adam.beta1());
  EXPECT_FLOAT_EQ(3, adam.beta2());
  EXPECT_FLOAT_EQ(4, adam.eps());
}

TEST_F(TrainerImplTest, CheckSGDSaveLoad) {
  SGD sgd(1);
  sgd.set_epoch(2);
  sgd.set_weight_decay(3);
  sgd.set_gradient_clipping(4);

  const std::string path = "/tmp/primitiv_TrainerImplTest_CheckSGDSaveLoad.data";
  sgd.save(path);

  std::shared_ptr<Trainer> loaded = Trainer::load(path);
  std::remove(path.c_str());

  ASSERT_EQ("SGD", loaded->name());

  std::shared_ptr<SGD> sgd2 = std::static_pointer_cast<SGD>(loaded);
  EXPECT_EQ(1, sgd2->eta());
  EXPECT_EQ(2, sgd2->get_epoch());
  EXPECT_EQ(3, sgd2->get_weight_decay());
  EXPECT_EQ(4, sgd2->get_gradient_clipping());
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

TEST_F(TrainerImplTest, CheckAdamSaveLoad) {
  Adam adam(1, 2, 3, 4);
  adam.set_epoch(5);
  adam.set_weight_decay(6);
  adam.set_gradient_clipping(7);

  const std::string path = "/tmp/primitiv_TrainerImplTest_CheckAdamSaveLoad.data";
  adam.save(path);

  std::shared_ptr<Trainer> loaded = Trainer::load(path);
  std::remove(path.c_str());

  ASSERT_EQ("Adam", loaded->name());

  std::shared_ptr<Adam> adam2 = std::static_pointer_cast<Adam>(loaded);
  EXPECT_EQ(1, adam2->alpha());
  EXPECT_EQ(2, adam2->beta1());
  EXPECT_EQ(3, adam2->beta2());
  EXPECT_EQ(4, adam2->eps());
  EXPECT_EQ(5, adam2->get_epoch());
  EXPECT_EQ(6, adam2->get_weight_decay());
  EXPECT_EQ(7, adam2->get_gradient_clipping());
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
