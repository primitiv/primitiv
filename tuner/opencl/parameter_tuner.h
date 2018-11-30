#ifndef PRIMITIV_TUNER_OPENCL_PARAMETER_TUNER_H_
#define PRIMITIV_TUNER_OPENCL_PARAMETER_TUNER_H_

#include <primitiv/primitiv.h>

#include <chrono>
#include <iostream>
#include <limits>
#include <vector>

namespace primitiv {

namespace devices {

class OpenCLParameterTuner {
public:
  OpenCLParameterTuner(
    std::string param_group_name, const std::vector<std::string> &param_names,
    std::size_t platform_id, std::size_t device_id, double min_calc_time,
    std::size_t iter_block_size)
  : param_group_name_(param_group_name)
  , param_names_(param_names)
  , platform_id_(platform_id)
  , device_id_(device_id)
  , min_calc_time_(min_calc_time)
  , iter_block_size_(iter_block_size) {}

  void tune(OpenCLKernelParameters &parameters) {
    std::vector<std::size_t> param_values(param_names_.size(), 1);
    OpenCL dev(platform_id_, device_id_, 0, parameters);
    dev.dump_description();
    double fastest_time = std::numeric_limits<double>::infinity();
    std::vector<std::size_t> fastest_param_values;
    std::cout << "Optimizing " << param_group_name_ << " ..." << std::endl;
    while (true) {
      for (std::size_t i = 0; i < param_names_.size(); ++i) {
        parameters.set_parameter(
          param_group_name_, param_names_[i], param_values[i]);
      }
      try {
        dev_.reset(
          new OpenCL(platform_id_, device_id_, 0, parameters));
        initialize();
        iter_function();
      } catch (...) {
        std::size_t d = param_values.size() - 1;
        for (; d >= 1; --d) {
          if (param_values[d] != 1) {
            param_values[d] = 1;
            param_values[d - 1] <<= 1;
            break;
          }
        }
        if (d == 0) {
          break;
        } else {
          continue;
        }
      }
      std::size_t iters_total = 0;
      auto start = std::chrono::high_resolution_clock::now();
      double elapsed = 0;
      while (elapsed < min_calc_time_) {
        for (std::size_t i = 0; i < iter_block_size_; ++i) {
          iter_function();
        }
        iters_total += iter_block_size_;
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      }
      const double time_mean = elapsed / iters_total;
      for (std::size_t i = 0; i < param_names_.size(); ++i) {
        std::cout << " " << param_names_[i] << "=" << param_values[i];
      }
      std::cout << ", time=" << time_mean << "[ms]" << std::endl;
      if (time_mean < fastest_time) {
        fastest_time = time_mean;
        fastest_param_values = param_values;
      }
      param_values.back() <<= 1;
    }
    for (std::size_t i = 0; i < param_names_.size(); ++i) {
      parameters.set_parameter(param_group_name_, param_names_[i], fastest_param_values[i]);
    }
    std::cout << " Fastest:";
    for (std::size_t i = 0; i < param_names_.size(); ++i) {
      std::cout << " " << param_names_[i] << "=" << fastest_param_values[i];
    }
    std::cout << std::endl;
  }

  virtual void initialize() {};
  virtual void iter_function() = 0;

private:
  const std::string param_group_name_;
  const std::vector<std::string> param_names_;
  const std::size_t platform_id_;
  const std::size_t device_id_;
  const double min_calc_time_;
  const std::size_t iter_block_size_;

protected:
  std::shared_ptr<Device> dev_;
};

}  // namespace devices

}  // namespace primitiv

#endif  // PRIMITIV_TUNER_OPENCL_PARAMETER_TUNER_H_
