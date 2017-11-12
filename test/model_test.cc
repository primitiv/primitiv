#include <config.h>

#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <primitiv/model.h>
#include <primitiv/parameter.h>

using std::map;
using std::string;
using std::vector;

namespace primitiv {

class ModelTest : public testing::Test {};

TEST_F(ModelTest, CheckAddParameter) {
  Model m;
  Parameter p1, p2, p3;

  EXPECT_NO_THROW(m.add_parameter("p1", p1));
  EXPECT_THROW(m.add_parameter("p1", p1), Error);
  EXPECT_THROW(m.add_parameter("x", p1), Error);

  EXPECT_THROW(m.add_parameter("p1", p2), Error);
  EXPECT_NO_THROW(m.add_parameter("p2", p2));
  EXPECT_THROW(m.add_parameter("p2", p2), Error);
  EXPECT_THROW(m.add_parameter("x", p2), Error);

  EXPECT_THROW(m.add_parameter("p1", p3), Error);
  EXPECT_THROW(m.add_parameter("p2", p3), Error);
  EXPECT_NO_THROW(m.add_parameter("p3", p3));
  EXPECT_THROW(m.add_parameter("p3", p3), Error);
  EXPECT_THROW(m.add_parameter("x", p3), Error);
}

TEST_F(ModelTest, CheckAddSubmodel) {
  Model m, sm1, sm2;
  Parameter p1, p2;

  EXPECT_NO_THROW(m.add_parameter("p1", p1));
  EXPECT_NO_THROW(m.add_submodel("sm1", sm1));
  EXPECT_THROW(m.add_submodel("sm1", sm1), Error);
  EXPECT_THROW(m.add_submodel("x", sm1), Error);

  EXPECT_THROW(m.add_parameter("p1", p2), Error);
  EXPECT_THROW(m.add_parameter("sm1", p2), Error);
  EXPECT_THROW(m.add_submodel("p1", sm2), Error);
  EXPECT_THROW(m.add_submodel("sm1", sm2), Error);

  EXPECT_NO_THROW(m.add_parameter("p2", p2));
  EXPECT_NO_THROW(m.add_submodel("sm2", sm2));
  EXPECT_THROW(m.add_submodel("sm2", sm2), Error);
  EXPECT_THROW(m.add_submodel("x", sm2), Error);
}

TEST_F(ModelTest, CheckAddSubmodelCycle) {
  Model m1, m2, m3, m4;

  EXPECT_THROW(m1.add_submodel("self", m1), Error);

  EXPECT_NO_THROW(m1.add_submodel("m2", m2));
  EXPECT_THROW(m2.add_submodel("m1", m1), Error);

  EXPECT_NO_THROW(m2.add_submodel("m3", m3));
  EXPECT_THROW(m3.add_submodel("m1", m1), Error);
  EXPECT_THROW(m3.add_submodel("m2", m2), Error);

  EXPECT_NO_THROW(m2.add_submodel("m4", m4));
  EXPECT_THROW(m4.add_submodel("m1", m1), Error);
  EXPECT_THROW(m4.add_submodel("m2", m2), Error);

  // NOTE(odashi):
  // This generates a diamond hierarchy.
  // We allow to do this for now, but Trainer.add_model() may fail because of
  // registering same parameters multiple times.
  EXPECT_NO_THROW(m4.add_submodel("m3", m3));
}

TEST_F(ModelTest, CheckGetParameter) {
  Model m, sm;
  Parameter p1, p2, p3;
  m.add_parameter("p1", p1);
  m.add_parameter("p2", p2);
  sm.add_parameter("p3", p3);
  m.add_submodel("sm", sm);

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
  m.add_parameter("p1", p1);
  sm.add_parameter("p2", p2);
  sm.add_parameter("p3", p3);
  m.add_submodel("sm", sm);

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
  m.add_parameter("p1", p1);
  sm.add_parameter("p2", p2);
  sm.add_parameter("p3", p3);
  m.add_submodel("sm", sm);

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
  m.add_parameter("p", p);
  m.add_submodel("sm1", sm1);
  m.add_submodel("sm2", sm2);
  sm1.add_submodel("ssm", ssm);

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
  m.add_parameter("p", p);
  m.add_submodel("sm1", sm1);
  m.add_submodel("sm2", sm2);
  sm1.add_submodel("ssm", ssm);

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
  m.add_parameter("p", p);
  m.add_submodel("sm1", sm1);
  m.add_submodel("sm2", sm2);
  sm1.add_submodel("ssm", ssm);

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
  m.add_parameter("p1", p1);
  m.add_parameter("p2", p2);
  m.add_parameter("p3", p3);
  const map<vector<string>, Parameter *> params = m.get_all_parameters();
  EXPECT_EQ(3u, params.size());
  EXPECT_EQ(&p1, params.at(vector<string> { "p1" }));
  EXPECT_EQ(&p2, params.at(vector<string> { "p2" }));
  EXPECT_EQ(&p3, params.at(vector<string> { "p3" }));
}

TEST_F(ModelTest, CheckGetAllParametersWithSubmodels) {
  Model m1, m2, m3;
  Parameter p1, p2, p3;
  m1.add_parameter("p", p1);
  m2.add_parameter("p", p2);
  m3.add_parameter("p", p3);
  m1.add_submodel("sm", m2);
  m2.add_submodel("sm", m3);

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
  m.add_parameter("p1", p1);
  m.add_parameter("p2", p2);
  m.add_parameter("p3", p3);
  const map<vector<string>, Parameter *> params = m.get_trainable_parameters();
  EXPECT_EQ(3u, params.size());
  EXPECT_EQ(&p1, params.at(vector<string> { "p1" }));
  EXPECT_EQ(&p2, params.at(vector<string> { "p2" }));
  EXPECT_EQ(&p3, params.at(vector<string> { "p3" }));
}

TEST_F(ModelTest, CheckGetTrainableParametersWithSubmodels) {
  Model m1, m2, m3;
  Parameter p1, p2, p3;
  m1.add_parameter("p", p1);
  m2.add_parameter("p", p2);
  m3.add_parameter("p", p3);
  m1.add_submodel("sm", m2);
  m2.add_submodel("sm", m3);

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

}  // namespace primitiv
