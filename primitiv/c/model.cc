#include <primitiv/config.h>

#include <string>
#include <vector>

#include <primitiv/model.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/model.h>

using primitiv::Model;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;

primitiv_Status primitiv_Model_new(primitiv_Model **model) try {
  *model = to_c_ptr(new Model());
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Model_delete(primitiv_Model *model) try {
  PRIMITIV_C_CHECK_NOT_NULL(model);
  delete to_cpp_ptr(model);
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Model_load(
    primitiv_Model *model, const char *path, PRIMITIV_C_BOOL with_stats,
    primitiv_Device *device) try {
  PRIMITIV_C_CHECK_NOT_NULL(model);
  PRIMITIV_C_CHECK_NOT_NULL(path);
  to_cpp_ptr(model)->load(path, with_stats, to_cpp_ptr(device));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Model_save(
    const primitiv_Model *model, const char *path,
    PRIMITIV_C_BOOL with_stats) try {
  PRIMITIV_C_CHECK_NOT_NULL(model);
  PRIMITIV_C_CHECK_NOT_NULL(path);
  to_cpp_ptr(model)->save(path, with_stats);
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Model_add_parameter(
    primitiv_Model *model, const char *name, primitiv_Parameter *param) try {
  PRIMITIV_C_CHECK_NOT_NULL(model);
  PRIMITIV_C_CHECK_NOT_NULL(name);
  to_cpp_ptr(model)->add(name, *to_cpp_ptr(param));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Model_add_model(
    primitiv_Model *model, const char *name, primitiv_Model *submodel) try {
  PRIMITIV_C_CHECK_NOT_NULL(model);
  PRIMITIV_C_CHECK_NOT_NULL(name);
  to_cpp_ptr(model)->add(name, *to_cpp_ptr(submodel));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Model_get_parameter(
    const primitiv_Model *model, const char **names, size_t n,
    const primitiv_Parameter **param) try {
  PRIMITIV_C_CHECK_NOT_NULL(model);
  PRIMITIV_C_CHECK_NOT_NULL(names);
  *param = to_c_ptr(&(to_cpp_ptr(model)->get_parameter(
      std::vector<std::string>(names, names + n))));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Model_get_submodel(
    const primitiv_Model *model, const char **names, size_t n,
    const primitiv_Model **submodel) try {
  PRIMITIV_C_CHECK_NOT_NULL(model);
  PRIMITIV_C_CHECK_NOT_NULL(names);
  *submodel = to_c_ptr(&(to_cpp_ptr(model)->get_submodel(
      std::vector<std::string>(names, names + n))));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
