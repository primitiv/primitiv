#ifndef PYTHON_PRIMITIV_PYTRAINER_H_
#define PYTHON_PRIMITIV_PYTRAINER_H_

#include <primitiv/trainer.h>
#include <iostream>

__PYX_EXTERN_C int python_primitiv_trainer_get_configs(
                        PyObject *obj,
                        std::unordered_map<std::string, unsigned> &uint_configs,
                        std::unordered_map<std::string, float> &float_configs);
__PYX_EXTERN_C int python_primitiv_trainer_set_configs(PyObject *obj,
                        const std::unordered_map<std::string, unsigned> &uint_configs,
                        const std::unordered_map<std::string, float> &float_configs);
__PYX_EXTERN_C int python_primitiv_trainer_configure_parameter(
                        PyObject *obj,
                        primitiv::Parameter &param);
__PYX_EXTERN_C int python_primitiv_trainer_update_parameter(
                        PyObject *obj,
                        float scale,
                        primitiv::Parameter &param);

namespace python_primitiv {

class PyTrainer : public primitiv::Trainer {

public:
  PyTrainer(PyObject *obj) : obj_(obj) {}

  void get_configs(std::unordered_map<std::string, unsigned> &uint_configs,
                   std::unordered_map<std::string, float> &float_configs) const override {
    int ret = python_primitiv_trainer_get_configs(obj_, uint_configs, float_configs);
    if (ret == -1) {
      // NOTE(vbkaisetsu): This is just a trigger of throwing an error.
      // This message is not passed to Python.
      THROW_ERROR("error: get_configs");
    }
    Trainer::get_configs(uint_configs, float_configs);
  }

  void set_configs(const std::unordered_map<std::string, unsigned> &uint_configs,
                   const std::unordered_map<std::string, float> &float_configs) override {
    Trainer::set_configs(uint_configs, float_configs);
    int ret = python_primitiv_trainer_set_configs(obj_, uint_configs, float_configs);
    if (ret == -1) {
      // NOTE(vbkaisetsu): This is just a trigger of throwing an error.
      // This message is not passed to Python.
      THROW_ERROR("error: set_configs");
    }
  }

  void configure_parameter(Parameter &param) override {
    int ret = python_primitiv_trainer_configure_parameter(obj_, param);
    if (ret == -1) {
      // NOTE(vbkaisetsu): This is just a trigger of throwing an error.
      // This message is not passed to Python.
      THROW_ERROR("error: configure_parameter");
    }
  }

  void update_parameter(float scale, Parameter &param) override {
    int ret = python_primitiv_trainer_update_parameter(obj_, scale, param);
    if (ret == -1) {
      // NOTE(vbkaisetsu): This is just a trigger of throwing an error.
      // This message is not passed to Python.
      THROW_ERROR("error: update_parameter");
    }
  }

private:
  PyObject *obj_;
};

}  // namespace python_primitiv

#endif  // PYTHON_PRIMITIV_PYTRAINER_H_
