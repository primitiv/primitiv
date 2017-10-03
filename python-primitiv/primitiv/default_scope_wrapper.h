#ifndef PYTHON_PRIMITIV_DEFAULT_SCOPE_WRAPPER_H_
#define PYTHON_PRIMITIV_DEFAULT_SCOPE_WRAPPER_H_

#include <primitiv/default_scope.h>

#include <primitiv/device.h>
#include <primitiv/graph.h>

namespace python_primitiv {

using namespace primitiv;

inline Device &DefaultScopeDevice_get() {
    return DefaultScope<Device>::get();
}

inline size_t DefaultScopeDevice_size() {
    return DefaultScope<Device>::size();
}

inline Graph &DefaultScopeGraph_get() {
    return DefaultScope<Graph>::get();
}

inline size_t DefaultScopeGraph_size() {
    return DefaultScope<Graph>::size();
}

}

#endif
