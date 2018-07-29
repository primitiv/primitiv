#include <primitiv/config.h>

#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <primitiv/core/model.h>
#include <primitiv/devices/naive/device.h>
#include <primitiv/core/parameter.h>

#include <test_utils.h>

using std::map;
using std::string;
using std::vector;
using test_utils::vector_match;

namespace primitiv {

class ModelTest : public testing::Test {
  devices::Naive dev;

protected:
  void SetUp() override {
    Device::set_default(dev);
  }
};

TEST_F(ModelTest, CheckAddParameter) {
  Model m;
  Parameter p1, p2, p3;

  EXPECT_NO_THROW(m.add("p1", p1));
  EXPECT_NO_THROW(m.add("p1", p1));
  EXPECT_THROW(m.add("x", p1), Error);

  EXPECT_THROW(m.add("p1", p2), Error);
  EXPECT_NO_THROW(m.add("p2", p2));
  EXPECT_NO_THROW(m.add("p2", p2));
  EXPECT_THROW(m.add("x", p2), Error);

  EXPECT_THROW(m.add("p1", p3), Error);
  EXPECT_THROW(m.add("p2", p3), Error);
  EXPECT_NO_THROW(m.add("p3", p3));
  EXPECT_NO_THROW(m.add("p3", p3));
  EXPECT_THROW(m.add("x", p3), Error);
}

TEST_F(ModelTest, CheckAddSubmodel) {
  Model m, sm1, sm2;
  Parameter p1, p2;

  EXPECT_NO_THROW(m.add("p1", p1));
  EXPECT_NO_THROW(m.add("sm1", sm1));
  EXPECT_NO_THROW(m.add("sm1", sm1));
  EXPECT_THROW(m.add("x", sm1), Error);

  EXPECT_THROW(m.add("p1", p2), Error);
  EXPECT_THROW(m.add("sm1", p2), Error);
  EXPECT_THROW(m.add("p1", sm2), Error);
  EXPECT_THROW(m.add("sm1", sm2), Error);

  EXPECT_NO_THROW(m.add("p2", p2));
  EXPECT_NO_THROW(m.add("sm2", sm2));
  EXPECT_NO_THROW(m.add("sm2", sm2));
  EXPECT_THROW(m.add("x", sm2), Error);
}

TEST_F(ModelTest, CheckAddSubmodelCycle) {
  Model m1, m2, m3, m4;

  EXPECT_THROW(m1.add("self", m1), Error);

  EXPECT_NO_THROW(m1.add("m2", m2));
  EXPECT_THROW(m2.add("m1", m1), Error);

  EXPECT_NO_THROW(m2.add("m3", m3));
  EXPECT_THROW(m3.add("m1", m1), Error);
  EXPECT_THROW(m3.add("m2", m2), Error);

  EXPECT_NO_THROW(m2.add("m4", m4));
  EXPECT_THROW(m4.add("m1", m1), Error);
  EXPECT_THROW(m4.add("m2", m2), Error);

  // Also allows a diamond hierarchy.
  EXPECT_NO_THROW(m4.add("m3", m3));
}

TEST_F(ModelTest, CheckGetParameter) {
  Model m, sm;
  Parameter p1, p2, p3;
  m.add("p1", p1);
  m.add("p2", p2);
  sm.add("p3", p3);
  m.add("sm", sm);

  EXPECT_EQ(&p1, &m.get_parameter("p1"));
  EXPECT_EQ(&p2, &m.get_parameter("p2"));
  EXPECT_THROW(m.get_parameter("p3"), Error);
  EXPECT_THROW(m.get_parameter("sm"), Error);
  EXPECT_THROW(m.get_parameter("x"), Error);

  const Model &rm = m;
  EXPECT_EQ(&p1, &rm.get_parameter("p1"));
  EXPECT_EQ(&p2, &rm.get_parameter("p2"));
  EXPECT_THROW(rm.get_parameter("p3"), Error);
  EXPECT_THROW(rm.get_parameter("sm"), Error);
  EXPECT_THROW(rm.get_parameter("x"), Error);
}

TEST_F(ModelTest, CheckGetParameterRecursiveByInitializerList) {
  Model m, sm;
  Parameter p1, p2, p3;
  m.add("p1", p1);
  sm.add("p2", p2);
  sm.add("p3", p3);
  m.add("sm", sm);

  EXPECT_EQ(&p1, &m.get_parameter({"p1"}));
  EXPECT_EQ(&p2, &m.get_parameter({"sm", "p2"}));
  EXPECT_EQ(&p3, &m.get_parameter({"sm", "p3"}));
  EXPECT_EQ(&p2, &sm.get_parameter({"p2"}));
  EXPECT_EQ(&p3, &sm.get_parameter({"p3"}));
  EXPECT_THROW(m.get_parameter({"p2"}), Error);
  EXPECT_THROW(m.get_parameter({"p3"}), Error);
  EXPECT_THROW(m.get_parameter({"sm"}), Error);
  EXPECT_THROW(m.get_parameter({"sm", "p1"}), Error);
  EXPECT_THROW(sm.get_parameter({"p1"}), Error);
  EXPECT_THROW(m.get_parameter({"x"}), Error);

  const Model &rm = m, &rsm = sm;
  EXPECT_EQ(&p1, &rm.get_parameter({"p1"}));
  EXPECT_EQ(&p2, &rm.get_parameter({"sm", "p2"}));
  EXPECT_EQ(&p3, &rm.get_parameter({"sm", "p3"}));
  EXPECT_EQ(&p2, &rsm.get_parameter({"p2"}));
  EXPECT_EQ(&p3, &rsm.get_parameter({"p3"}));
  EXPECT_THROW(rm.get_parameter({"p2"}), Error);
  EXPECT_THROW(rm.get_parameter({"p3"}), Error);
  EXPECT_THROW(rm.get_parameter({"sm"}), Error);
  EXPECT_THROW(rm.get_parameter({"sm", "p1"}), Error);
  EXPECT_THROW(rsm.get_parameter({"p1"}), Error);
  EXPECT_THROW(rm.get_parameter({"x"}), Error);
}

TEST_F(ModelTest, CheckGetParameterRecursiveByVector) {
  Model m, sm;
  Parameter p1, p2, p3;
  m.add("p1", p1);
  sm.add("p2", p2);
  sm.add("p3", p3);
  m.add("sm", sm);

  EXPECT_EQ(&p1, &m.get_parameter(vector<string> {"p1"}));
  EXPECT_EQ(&p2, &m.get_parameter(vector<string> {"sm", "p2"}));
  EXPECT_EQ(&p3, &m.get_parameter(vector<string> {"sm", "p3"}));
  EXPECT_EQ(&p2, &sm.get_parameter(vector<string> {"p2"}));
  EXPECT_EQ(&p3, &sm.get_parameter(vector<string> {"p3"}));
  EXPECT_THROW(m.get_parameter(vector<string> {"p2"}), Error);
  EXPECT_THROW(m.get_parameter(vector<string> {"p3"}), Error);
  EXPECT_THROW(m.get_parameter(vector<string> {"sm"}), Error);
  EXPECT_THROW(m.get_parameter(vector<string> {"sm", "p1"}), Error);
  EXPECT_THROW(sm.get_parameter(vector<string> {"p1"}), Error);
  EXPECT_THROW(m.get_parameter(vector<string> {"x"}), Error);

  const Model &rm = m, &rsm = sm;
  EXPECT_EQ(&p1, &rm.get_parameter(vector<string> {"p1"}));
  EXPECT_EQ(&p2, &rm.get_parameter(vector<string> {"sm", "p2"}));
  EXPECT_EQ(&p3, &rm.get_parameter(vector<string> {"sm", "p3"}));
  EXPECT_EQ(&p2, &rsm.get_parameter(vector<string> {"p2"}));
  EXPECT_EQ(&p3, &rsm.get_parameter(vector<string> {"p3"}));
  EXPECT_THROW(rm.get_parameter(vector<string> {"p2"}), Error);
  EXPECT_THROW(rm.get_parameter(vector<string> {"p3"}), Error);
  EXPECT_THROW(rm.get_parameter(vector<string> {"sm"}), Error);
  EXPECT_THROW(rm.get_parameter(vector<string> {"sm", "p1"}), Error);
  EXPECT_THROW(rsm.get_parameter(vector<string> {"p1"}), Error);
  EXPECT_THROW(rm.get_parameter(vector<string> {"x"}), Error);
}

TEST_F(ModelTest, CheckGetSubmodel) {
  Model m, sm1, sm2, ssm;
  Parameter p;
  m.add("p", p);
  m.add("sm1", sm1);
  m.add("sm2", sm2);
  sm1.add("ssm", ssm);

  EXPECT_EQ(&sm1, &m.get_submodel("sm1"));
  EXPECT_EQ(&sm2, &m.get_submodel("sm2"));
  EXPECT_THROW(m.get_submodel("ssm"), Error);
  EXPECT_THROW(m.get_submodel("p"), Error);

  const Model &rm = m;
  EXPECT_EQ(&sm1, &rm.get_submodel("sm1"));
  EXPECT_EQ(&sm2, &rm.get_submodel("sm2"));
  EXPECT_THROW(rm.get_submodel("ssm"), Error);
  EXPECT_THROW(rm.get_submodel("p"), Error);
}

TEST_F(ModelTest, CheckGetSubmodelRecursiveByInitializerList) {
  Model m, sm1, sm2, ssm;
  Parameter p;
  m.add("p", p);
  m.add("sm1", sm1);
  m.add("sm2", sm2);
  sm1.add("ssm", ssm);

  EXPECT_EQ(&sm1, &m.get_submodel({"sm1"}));
  EXPECT_EQ(&sm2, &m.get_submodel({"sm2"}));
  EXPECT_EQ(&ssm, &m.get_submodel({"sm1", "ssm"}));
  EXPECT_EQ(&ssm, &sm1.get_submodel({"ssm"}));
  EXPECT_THROW(m.get_submodel({"p"}), Error);
  EXPECT_THROW(m.get_submodel({"ssm"}), Error);
  EXPECT_THROW(m.get_submodel({"sm2", "ssm"}), Error);
  EXPECT_THROW(m.get_submodel({"x"}), Error);

  const Model &rm = m, &rsm1 = sm1;
  EXPECT_EQ(&sm1, &rm.get_submodel({"sm1"}));
  EXPECT_EQ(&sm2, &rm.get_submodel({"sm2"}));
  EXPECT_EQ(&ssm, &rm.get_submodel({"sm1", "ssm"}));
  EXPECT_EQ(&ssm, &rsm1.get_submodel({"ssm"}));
  EXPECT_THROW(rm.get_submodel({"p"}), Error);
  EXPECT_THROW(rm.get_submodel({"ssm"}), Error);
  EXPECT_THROW(rm.get_submodel({"sm2", "ssm"}), Error);
  EXPECT_THROW(rm.get_submodel({"x"}), Error);
}

TEST_F(ModelTest, CheckGetSubmodelRecursiveByVector) {
  Model m, sm1, sm2, ssm;
  Parameter p;
  m.add("p", p);
  m.add("sm1", sm1);
  m.add("sm2", sm2);
  sm1.add("ssm", ssm);

  EXPECT_EQ(&sm1, &m.get_submodel(vector<string> {"sm1"}));
  EXPECT_EQ(&sm2, &m.get_submodel(vector<string> {"sm2"}));
  EXPECT_EQ(&ssm, &m.get_submodel(vector<string> {"sm1", "ssm"}));
  EXPECT_EQ(&ssm, &sm1.get_submodel(vector<string> {"ssm"}));
  EXPECT_THROW(m.get_submodel(vector<string> {"p"}), Error);
  EXPECT_THROW(m.get_submodel(vector<string> {"ssm"}), Error);
  EXPECT_THROW(m.get_submodel(vector<string> {"sm2", "ssm"}), Error);
  EXPECT_THROW(m.get_submodel(vector<string> {"x"}), Error);

  const Model &rm = m, &rsm1 = sm1;
  EXPECT_EQ(&sm1, &rm.get_submodel(vector<string> {"sm1"}));
  EXPECT_EQ(&sm2, &rm.get_submodel(vector<string> {"sm2"}));
  EXPECT_EQ(&ssm, &rm.get_submodel(vector<string> {"sm1", "ssm"}));
  EXPECT_EQ(&ssm, &rsm1.get_submodel(vector<string> {"ssm"}));
  EXPECT_THROW(rm.get_submodel(vector<string> {"p"}), Error);
  EXPECT_THROW(rm.get_submodel(vector<string> {"ssm"}), Error);
  EXPECT_THROW(rm.get_submodel(vector<string> {"sm2", "ssm"}), Error);
  EXPECT_THROW(rm.get_submodel(vector<string> {"x"}), Error);
}

TEST_F(ModelTest, CheckGetAllParameters) {
  Model m;
  Parameter p1, p2, p3;
  m.add("p1", p1);
  m.add("p2", p2);
  m.add("p3", p3);
  const map<vector<string>, Parameter *> params = m.get_all_parameters();
  EXPECT_EQ(3u, params.size());
  EXPECT_EQ(&p1, params.at(vector<string> { "p1" }));
  EXPECT_EQ(&p2, params.at(vector<string> { "p2" }));
  EXPECT_EQ(&p3, params.at(vector<string> { "p3" }));
}

TEST_F(ModelTest, CheckGetAllParametersWithSubmodels) {
  Model m1, m2, m3;
  Parameter p1, p2, p3;
  m1.add("p", p1);
  m2.add("p", p2);
  m3.add("p", p3);
  m1.add("sm", m2);
  m2.add("sm", m3);

  const map<vector<string>, Parameter *> params1 = m1.get_all_parameters();
  EXPECT_EQ(3u, params1.size());
  EXPECT_EQ(&p1, params1.at(vector<string> { "p" }));
  EXPECT_EQ(&p2, params1.at(vector<string> { "sm", "p" }));
  EXPECT_EQ(&p3, params1.at(vector<string> { "sm", "sm", "p" }));

  const map<vector<string>, Parameter *> params2 = m2.get_all_parameters();
  EXPECT_EQ(2u, params2.size());
  EXPECT_EQ(&p2, params2.at(vector<string> { "p" }));
  EXPECT_EQ(&p3, params2.at(vector<string> { "sm", "p" }));

  const map<vector<string>, Parameter *> params3 = m3.get_all_parameters();
  EXPECT_EQ(1u, params3.size());
  EXPECT_EQ(&p3, params3.at(vector<string> { "p" }));
}

TEST_F(ModelTest, CheckGetTrainableParameters) {
  Model m;
  Parameter p1, p2, p3;
  m.add("p1", p1);
  m.add("p2", p2);
  m.add("p3", p3);
  const map<vector<string>, Parameter *> params = m.get_trainable_parameters();
  EXPECT_EQ(3u, params.size());
  EXPECT_EQ(&p1, params.at(vector<string> { "p1" }));
  EXPECT_EQ(&p2, params.at(vector<string> { "p2" }));
  EXPECT_EQ(&p3, params.at(vector<string> { "p3" }));
}

TEST_F(ModelTest, CheckGetTrainableParametersWithSubmodels) {
  Model m1, m2, m3;
  Parameter p1, p2, p3;
  m1.add("p", p1);
  m2.add("p", p2);
  m3.add("p", p3);
  m1.add("sm", m2);
  m2.add("sm", m3);

  const map<vector<string>, Parameter *> params1 = m1.get_trainable_parameters();
  EXPECT_EQ(3u, params1.size());
  EXPECT_EQ(&p1, params1.at(vector<string> { "p" }));
  EXPECT_EQ(&p2, params1.at(vector<string> { "sm", "p" }));
  EXPECT_EQ(&p3, params1.at(vector<string> { "sm", "sm", "p" }));

  const map<vector<string>, Parameter *> params2 = m2.get_trainable_parameters();
  EXPECT_EQ(2u, params2.size());
  EXPECT_EQ(&p2, params2.at(vector<string> { "p" }));
  EXPECT_EQ(&p3, params2.at(vector<string> { "sm", "p" }));

  const map<vector<string>, Parameter *> params3 = m3.get_trainable_parameters();
  EXPECT_EQ(1u, params3.size());
  EXPECT_EQ(&p3, params3.at(vector<string> { "p" }));
}

TEST_F(ModelTest, CheckSaveLoad_Same) {
  const Shape shape {2, 2};
  const vector<float> values1 {1, 2, 3, 4};
  const vector<float> values2 {5, 6, 7, 8};
  const string path = "/tmp/primitiv_ModelTest_CheckSaveLoad_Same.data";

  {
    Model m1, m2;
    Parameter p1(shape, values1), p2(shape, values2);
    m1.add("p", p1);
    m2.add("p", p2);
    m1.add("sm", m2);

    ASSERT_NO_THROW(m1.save(path));
  }

  {
    Model m1, m2;
    Parameter p1, p2;
    m1.add("p", p1);
    m2.add("p", p2);
    m1.add("sm", m2);

    EXPECT_NO_THROW(m1.load(path));
    std::remove(path.c_str());

    ASSERT_TRUE(p1.valid());
    ASSERT_TRUE(p2.valid());
    EXPECT_EQ(shape, p1.shape());
    EXPECT_EQ(shape, p2.shape());
    EXPECT_TRUE(vector_match(values1, p1.value().to_vector()));
    EXPECT_TRUE(vector_match(values2, p2.value().to_vector()));
  }
}

TEST_F(ModelTest, CheckSaveLoad_Insufficient) {
  const Shape shape {2, 2};
  const vector<float> values1 {1, 2, 3, 4};
  const vector<float> values2 {5, 6, 7, 8};
  const string path = "/tmp/primitiv_ModelTest_CheckSaveLoad_Insufficient.data";

  {
    Model m1, m2;
    Parameter p1(shape, values1), p2(shape, values2);
    m1.add("p", p1);
    m2.add("p", p2);
    m1.add("sm", m2);

    ASSERT_NO_THROW(m1.save(path));
  }

  {
    Model m1, m2;
    Parameter p1;
    m1.add("p", p1);
    m1.add("sm", m2);

    EXPECT_THROW(m1.load(path), Error);
    std::remove(path.c_str());
  }
}

TEST_F(ModelTest, CheckSaveLoad_Excessive) {
  const Shape shape {2, 2};
  const vector<float> values1 {1, 2, 3, 4};
  const vector<float> values2 {5, 6, 7, 8};
  const string path = "/tmp/primitiv_ModelTest_CheckSaveLoad_Excessive.data";

  {
    Model m1, m2;
    Parameter p1(shape, values1), p2(shape, values2);
    m1.add("p", p1);
    m2.add("p", p2);
    m1.add("sm", m2);

    ASSERT_NO_THROW(m1.save(path));
  }

  {
    Model m1, m2;
    Parameter p1, p2, p3;
    m1.add("p", p1);
    m2.add("p", p2);
    m2.add("pp", p3);
    m1.add("sm", m2);

    EXPECT_NO_THROW(m1.load(path));
    std::remove(path.c_str());

    ASSERT_TRUE(p1.valid());
    ASSERT_TRUE(p2.valid());
    ASSERT_FALSE(p3.valid());
    EXPECT_EQ(shape, p1.shape());
    EXPECT_EQ(shape, p2.shape());
    EXPECT_TRUE(vector_match(values1, p1.value().to_vector()));
    EXPECT_TRUE(vector_match(values2, p2.value().to_vector()));
  }
}

TEST_F(ModelTest, CheckSaveLoadWithStats) {
  const Shape shape {2, 2};
  const vector<float> values1 {1, 2, 3, 4};
  const vector<float> values2 {5, 6, 7, 8};
  const vector<float> stats1 {10, 20, 30, 40};
  const vector<float> stats2 {50, 60, 70, 80};
  const string path = "/tmp/primitiv_ModelTest_CheckSaveLoadWithStats.data";

  {
    Model m1, m2;
    Parameter p1(shape, values1), p2(shape, values2);
    p1.add_stats("a", shape);
    p2.add_stats("b", shape);
    p1.stats("a").reset_by_vector(stats1);
    p2.stats("b").reset_by_vector(stats2);
    m1.add("p", p1);
    m2.add("p", p2);
    m1.add("sm", m2);

    ASSERT_NO_THROW(m1.save(path));
  }

  {
    Model m1, m2;
    Parameter p1, p2;
    m1.add("p", p1);
    m2.add("p", p2);
    m1.add("sm", m2);

    EXPECT_NO_THROW(m1.load(path));
    std::remove(path.c_str());

    ASSERT_TRUE(p1.valid());
    ASSERT_TRUE(p2.valid());
    EXPECT_EQ(shape, p1.shape());
    EXPECT_EQ(shape, p2.shape());
    EXPECT_TRUE(vector_match(values1, p1.value().to_vector()));
    EXPECT_TRUE(vector_match(values2, p2.value().to_vector()));
    ASSERT_TRUE(p1.has_stats("a"));
    ASSERT_TRUE(p2.has_stats("b"));
    EXPECT_TRUE(vector_match(stats1, p1.stats("a").to_vector()));
    EXPECT_TRUE(vector_match(stats2, p2.stats("b").to_vector()));
  }
}

TEST_F(ModelTest, CheckSaveWithoutStats) {
  const Shape shape {2, 2};
  const vector<float> values1 {1, 2, 3, 4};
  const vector<float> values2 {5, 6, 7, 8};
  const vector<float> stats1 {10, 20, 30, 40};
  const vector<float> stats2 {50, 60, 70, 80};
  const string path = "/tmp/primitiv_ModelTest_CheckSaveWithoutStats.data";

  {
    Model m1, m2;
    Parameter p1(shape, values1), p2(shape, values2);
    p1.add_stats("a", shape);
    p2.add_stats("b", shape);
    p1.stats("a").reset_by_vector(stats1);
    p2.stats("b").reset_by_vector(stats2);
    m1.add("p", p1);
    m2.add("p", p2);
    m1.add("sm", m2);

    ASSERT_NO_THROW(m1.save(path, false));
  }

  {
    Model m1, m2;
    Parameter p1, p2;
    m1.add("p", p1);
    m2.add("p", p2);
    m1.add("sm", m2);

    EXPECT_NO_THROW(m1.load(path));
    std::remove(path.c_str());

    ASSERT_TRUE(p1.valid());
    ASSERT_TRUE(p2.valid());
    EXPECT_EQ(shape, p1.shape());
    EXPECT_EQ(shape, p2.shape());
    EXPECT_TRUE(vector_match(values1, p1.value().to_vector()));
    EXPECT_TRUE(vector_match(values2, p2.value().to_vector()));
    EXPECT_FALSE(p1.has_stats("a"));
    EXPECT_FALSE(p2.has_stats("b"));
  }
}

TEST_F(ModelTest, CheckLoadWithoutStats) {
  const Shape shape {2, 2};
  const vector<float> values1 {1, 2, 3, 4};
  const vector<float> values2 {5, 6, 7, 8};
  const vector<float> stats1 {10, 20, 30, 40};
  const vector<float> stats2 {50, 60, 70, 80};
  const string path = "/tmp/primitiv_ModelTest_CheckLoadWithoutStats.data";

  {
    Model m1, m2;
    Parameter p1(shape, values1), p2(shape, values2);
    p1.add_stats("a", shape);
    p2.add_stats("b", shape);
    p1.stats("a").reset_by_vector(stats1);
    p2.stats("b").reset_by_vector(stats2);
    m1.add("p", p1);
    m2.add("p", p2);
    m1.add("sm", m2);

    ASSERT_NO_THROW(m1.save(path));
  }

  {
    Model m1, m2;
    Parameter p1, p2;
    m1.add("p", p1);
    m2.add("p", p2);
    m1.add("sm", m2);

    EXPECT_NO_THROW(m1.load(path, false));
    std::remove(path.c_str());

    ASSERT_TRUE(p1.valid());
    ASSERT_TRUE(p2.valid());
    EXPECT_EQ(shape, p1.shape());
    EXPECT_EQ(shape, p2.shape());
    EXPECT_TRUE(vector_match(values1, p1.value().to_vector()));
    EXPECT_TRUE(vector_match(values2, p2.value().to_vector()));
    EXPECT_FALSE(p1.has_stats("a"));
    EXPECT_FALSE(p2.has_stats("b"));
  }
}

}  // namespace primitiv
