#include <config.h>

#include <vector>
#include <gtest/gtest.h>
#include <primitiv/model.h>
#include <primitiv/parameter.h>

using std::set;
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

TEST_F(ModelTest, CheckGetTrainableParameters) {
  Model m;
  Parameter p1, p2, p3;
  m.add_parameter("p1", p1);
  m.add_parameter("p2", p2);
  m.add_parameter("p3", p3);
  const vector<Parameter *> params = m.get_trainable_parameters();
  EXPECT_EQ(3u, params.size());
  EXPECT_NE(params.end(), std::find(params.begin(), params.end(), &p1));
  EXPECT_NE(params.end(), std::find(params.begin(), params.end(), &p2));
  EXPECT_NE(params.end(), std::find(params.begin(), params.end(), &p3));
}

TEST_F(ModelTest, CheckGetTrainableParametersWithSubmodels) {
  Model m1, m2, m3;
  Parameter p1, p2, p3;
  m1.add_parameter("p", p1);
  m2.add_parameter("p", p2);
  m3.add_parameter("p", p3);
  m1.add_submodel("sm", m2);
  m2.add_submodel("sm", m3);

  const vector<Parameter *> params1 = m1.get_trainable_parameters();
  EXPECT_EQ(3u, params1.size());
  EXPECT_NE(params1.end(), std::find(params1.begin(), params1.end(), &p1));
  EXPECT_NE(params1.end(), std::find(params1.begin(), params1.end(), &p2));
  EXPECT_NE(params1.end(), std::find(params1.begin(), params1.end(), &p3));

  const vector<Parameter *> params2 = m2.get_trainable_parameters();
  EXPECT_EQ(2u, params2.size());
  EXPECT_NE(params2.end(), std::find(params2.begin(), params2.end(), &p2));
  EXPECT_NE(params2.end(), std::find(params2.begin(), params2.end(), &p3));

  const vector<Parameter *> params3 = m3.get_trainable_parameters();
  EXPECT_EQ(1u, params3.size());
  EXPECT_NE(params3.end(), std::find(params3.begin(), params3.end(), &p3));
}

}  // namespace primitiv
