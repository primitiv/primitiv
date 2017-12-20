/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <string>
#include <vector>

#include <primitiv/model.h>

#include <primitiv/c/internal.h>
#include <primitiv/c/model.h>

using primitiv::Model;

extern "C" {

primitiv_Model *primitiv_Model_new() {
  return to_c(new Model());
}
primitiv_Model *safe_primitiv_Model_new(primitiv_Status *status) {
  SAFE_RETURN(primitiv_Model_new(), status, nullptr);
}

void primitiv_Model_delete(primitiv_Model *model) {
  delete to_cc(model);
}
void safe_primitiv_Model_delete(primitiv_Model *model,
                                primitiv_Status *status) {
  SAFE_EXPR(primitiv_Model_delete(model), status);
}

void primitiv_Model_load(primitiv_Model *model,
                         const char *path,
                         bool with_stats,
                         primitiv_Device *device) {
  to_cc(model)->load(path, with_stats, to_cc(device));
}
void safe_primitiv_Model_load(primitiv_Model *model,
                              const char *path,
                              bool with_stats,
                              primitiv_Device *device,
                              primitiv_Status *status) {
  SAFE_EXPR(primitiv_Model_load(model, path, with_stats, device), status);
}

void primitiv_Model_save(const primitiv_Model *model,
                         const char *path,
                         bool with_stats) {
  to_cc(model)->save(path, with_stats);
}
void safe_primitiv_Model_save(const primitiv_Model *model,
                              const char *path,
                              bool with_stats,
                              primitiv_Status *status) {
  SAFE_EXPR(primitiv_Model_save(model, path, with_stats), status);
}

void primitiv_Model_add_parameter(primitiv_Model *model,
                                  const char *name,
                                  primitiv_Parameter *param) {
  to_cc(model)->add(name, *to_cc(param));
}
void safe_primitiv_Model_add_parameter(primitiv_Model *model,
                                       const char *name,
                                       primitiv_Parameter *param,
                                       primitiv_Status *status) {
  SAFE_EXPR(primitiv_Model_add_parameter(model, name, param), status);
}

void primitiv_Model_add_model(primitiv_Model *self,
                              const char *name,
                              primitiv_Model *param) {
  to_cc(self)->add(name, *to_cc(param));
}
void safe_primitiv_Model_add_model(primitiv_Model *self,
                                               const char *name,
                                               primitiv_Model *model,
                                               primitiv_Status *status) {
  SAFE_EXPR(primitiv_Model_add_model(self, name, model), status);
}

const primitiv_Parameter *primitiv_Model_get_parameter(
    const primitiv_Model *model,
    const char **names,
    size_t n) {
  return to_c(&(to_cc(model)->get_parameter(
      std::vector<std::string>(names, names + n))));
}
const primitiv_Parameter *safe_primitiv_Model_get_parameter(
    const primitiv_Model *model,
    const char **names,
    size_t n,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_Model_get_parameter(model, names, n), status, nullptr);
}

const primitiv_Model *primitiv_Model_get_submodel(
    const primitiv_Model *model,
    const char **names,
    size_t n) {
  return to_c(&(to_cc(model)->get_submodel(
      std::vector<std::string>(names, names + n))));
}
const primitiv_Model *safe_primitiv_Model_get_submodel(
    const primitiv_Model *model,
    const char **names,
    size_t n,
    primitiv_Status *status) {
  SAFE_RETURN(primitiv_Model_get_submodel(model, names, n), status, nullptr);
}

}  // end extern "C"
