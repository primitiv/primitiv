#include "primitiv_c/internal.h"
#include "primitiv_c/parameter.h"

#include <vector>

#include <primitiv/parameter.h>

using primitiv::Parameter;

extern "C" {

primitiv_Parameter *primitiv_Parameter_new() {
  return to_c(new Parameter());
}

primitiv_Parameter *primitiv_Parameter_new_with_values(
    const primitiv_Shape *shape,
    const float *value,
    size_t n,
    primitiv_Device *device) {
  if (!device) {
    device = primitiv_Device_get_default();
  }
  return to_c(new Parameter(*to_cc(shape),
                            std::vector<float>(value, value + n),
                            *to_cc(device)));
}

primitiv_Parameter *primitiv_Parameter_new_with_initializer(
    const primitiv_Shape *shape,
    const primitiv_Initializer *initializer,
    primitiv_Device *device) {
  if (!device) {
    device = primitiv_Device_get_default();
  }
  return to_c(new Parameter(*to_cc(shape),
                            *to_cc(initializer),
                            *to_cc(device)));
}

void primitiv_Parameter_delete(primitiv_Parameter *parameter) {
  delete to_cc(parameter);
}

void primitiv_Parameter_init_with_values(
    primitiv_Parameter *parameter,
    const primitiv_Shape *shape,
    const float *value,
    size_t n,
    primitiv_Device *device) {
  if (!device) {
    device = primitiv_Device_get_default();
  }
  to_cc(parameter)->init(*to_cc(shape),
                         std::vector<float>(value, value + n),
                         *to_cc(device));
}

void primitiv_Parameter_init_with_initializer(
    primitiv_Parameter *parameter,
    const primitiv_Shape *shape,
    const primitiv_Initializer *initializer,
    primitiv_Device *device) {
  if (!device) {
    device = primitiv_Device_get_default();
  }
  to_cc(parameter)->init(*to_cc(shape),
                         *to_cc(initializer),
                         *to_cc(device));
}

void primitiv_Parameter_load(
    primitiv_Parameter *parameter,
    const char *path,
    bool with_stats,
    primitiv_Device *device) {
  if (!device) {
    device = primitiv_Device_get_default();
  }
  to_cc(parameter)->load(path, with_stats, *to_cc(device));
}

void primitiv_Parameter_save(
    const primitiv_Parameter *parameter,
    const char *path,
    bool with_stats) {
  to_cc(parameter)->save(path, with_stats);
}

bool primitiv_Parameter_valid(const primitiv_Parameter *parameter) {
  return to_cc(parameter)->valid();
}

void primitiv_Parameter_reset_gradients(primitiv_Parameter *parameter) {
  to_cc(parameter)->reset_gradient();
}

void primitiv_Parameter_add_stats(
    primitiv_Parameter *parameter,
    const char *name,
    const primitiv_Shape *shape) {
  to_cc(parameter)->add_stats(name, *to_cc(shape));
}

bool primitiv_Parameter_has_stats(
    primitiv_Parameter *parameter,
    const char *name) {
  return to_cc(parameter)->has_stats(name);
}

primitiv_Shape *primitiv_Parameter_shape(const primitiv_Parameter *parameter) {
  return to_c_from_value(to_cc(parameter)->shape());
}

primitiv_Device *primitiv_Parameter_device(const primitiv_Parameter *parameter) {
  return to_c(&to_cc(parameter)->device());
}

const primitiv_Tensor *primitiv_Parameter_value(const primitiv_Parameter *parameter) {
  return to_c(&to_cc(parameter)->value());
}

const primitiv_Tensor *primitiv_Parameter_gradient(const primitiv_Parameter *parameter) {
  return to_c(&to_cc(parameter)->gradient());
}

const primitiv_Tensor *primitiv_Parameter_stats(const primitiv_Parameter *parameter, const char *name) {
  return to_c(&to_cc(parameter)->stats(name));
}

}  // end extern "C"
