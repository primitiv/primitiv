/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <config.h>

#include <string>
#include <vector>

#include <primitiv/model.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/model.h>

using primitiv::Model;
using primitiv::c::internal::to_c;
using primitiv::c::internal::to_cc;

extern "C" {

primitiv_Model *primitiv_Model_new() {
  return to_c(new Model());
}

void primitiv_Model_delete(primitiv_Model *model) {
  delete to_cc(model);
}

primitiv_Status primitiv_Model_load(primitiv_Model *model, const char *path,
                                    bool with_stats, primitiv_Device *device) {
  try {
    to_cc(model)->load(path, with_stats, to_cc(device));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Model_save(const primitiv_Model *model,
                                    const char *path, bool with_stats) {
  try {
    to_cc(model)->save(path, with_stats);
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Model_add_parameter(
    primitiv_Model *model, const char *name, primitiv_Parameter *param) {
  try {
    to_cc(model)->add(name, *to_cc(param));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Model_add_model(
    primitiv_Model *model, const char *name, primitiv_Model *submodel) {
  try {
    to_cc(model)->add(name, *to_cc(submodel));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Model_get_parameter(
    const primitiv_Model *model, const char **names, size_t n,
    const primitiv_Parameter **param) {
  try {
    *param = to_c(&(to_cc(model)->get_parameter(
        std::vector<std::string>(names, names + n))));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_Model_get_submodel(
    const primitiv_Model *model, const char **names, size_t n,
    const primitiv_Model **submodel) {
  try {
    *submodel = to_c(&(to_cc(model)->get_submodel(
        std::vector<std::string>(names, names + n))));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

}  // end extern "C"
