#include <primitiv/config.h>

#include <string>
#include <vector>

#include <primitiv/core/model.h>
#include <primitiv/c/internal/internal.h>
#include <primitiv/c/model.h>

using primitiv::Model;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;

PRIMITIV_C_STATUS primitivCreateModel(primitivModel_t **newobj) try {
  PRIMITIV_C_CHECK_NOT_NULL(newobj);
  *newobj = to_c_ptr(new Model());
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivDeleteModel(primitivModel_t *model) try {
  PRIMITIV_C_CHECK_NOT_NULL(model);
  delete to_cpp_ptr(model);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivLoadModel(
    primitivModel_t *model, const char *path, PRIMITIV_C_BOOL with_stats,
    primitivDevice_t *device) try {
  PRIMITIV_C_CHECK_NOT_NULL(model);
  PRIMITIV_C_CHECK_NOT_NULL(path);
  to_cpp_ptr(model)->load(path, with_stats, to_cpp_ptr(device));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivSaveModel(
    const primitivModel_t *model, const char *path,
    PRIMITIV_C_BOOL with_stats) try {
  PRIMITIV_C_CHECK_NOT_NULL(model);
  PRIMITIV_C_CHECK_NOT_NULL(path);
  to_cpp_ptr(model)->save(path, with_stats);
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivAddParameterToModel(
    primitivModel_t *model, const char *name, primitivParameter_t *param) try {
  PRIMITIV_C_CHECK_NOT_NULL(model);
  PRIMITIV_C_CHECK_NOT_NULL(name);
  PRIMITIV_C_CHECK_NOT_NULL(param);
  to_cpp_ptr(model)->add(name, *to_cpp_ptr(param));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivAddSubmodelToModel(
    primitivModel_t *model, const char *name, primitivModel_t *submodel) try {
  PRIMITIV_C_CHECK_NOT_NULL(model);
  PRIMITIV_C_CHECK_NOT_NULL(name);
  PRIMITIV_C_CHECK_NOT_NULL(submodel);
  to_cpp_ptr(model)->add(name, *to_cpp_ptr(submodel));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetParameterFromModel(
    const primitivModel_t *model, const char **names, size_t n,
    const primitivParameter_t **retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(model);
  PRIMITIV_C_CHECK_NOT_NULL(names);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_c_ptr(&(to_cpp_ptr(model)->get_parameter(
      std::vector<std::string>(names, names + n))));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

PRIMITIV_C_STATUS primitivGetSubmodelFromModel(
    const primitivModel_t *model, const char **names, size_t n,
    const primitivModel_t **retval) try {
  PRIMITIV_C_CHECK_NOT_NULL(model);
  PRIMITIV_C_CHECK_NOT_NULL(names);
  PRIMITIV_C_CHECK_NOT_NULL(retval);
  *retval = to_c_ptr(&(to_cpp_ptr(model)->get_submodel(
      std::vector<std::string>(names, names + n))));
  return PRIMITIV_C_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS
