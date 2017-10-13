#ifndef PYTHON_PRIMITIV_PARAMETER_LOAD_WRAPPER_H_
#define PYTHON_PRIMITIV_PARAMETER_LOAD_WRAPPER_H_

#include <primitiv/parameter.h>
#include <primitiv/device.h>

namespace python_primitiv {

using namespace primitiv;
using namespace std;

inline Parameter* Parameter_load(string path, bool with_stats, Device &device) {
    return new Parameter(Parameter::load(path, with_stats, device));
}

}  // namespace python_primitiv

#endif  // PYTHON_PRIMITIV_PARAMETER_LOAD_WRAPPER_H_
