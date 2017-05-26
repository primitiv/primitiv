set(primitiv_LIBRARY_HDRS
  constant_initializer.h
  cpu_device.h
  device.h
  function.h
  function_impl.h
  graph.h
  initializer.h
  node.h
  node_ops.h
  parameter.h
  sgd_trainer.h
  shape.h
  tensor.h
  tensor_ops.h
  trainer.h
)

set(primitiv_LIBRARY_SRCS
  constant_initializer.cc
  cpu_device.cc
  function_impl.cc
  graph.cc
  parameter.cc
  sgd_trainer.cc
  shape.cc
  tensor.cc
)

add_library(primitiv SHARED ${primitiv_LIBRARY_HDRS} ${primitiv_LIBRARY_SRCS})
target_link_libraries(primitiv)

if(USE_CUDA)
  set(primitiv_cuda_LIBRARY_HDRS ${primitiv_LIBRARY_HDRS} cuda_device.h)
  set(primitiv_cuda_LIBRARY_SRCS cuda_device.cu)
  cuda_add_library(primitiv_cuda SHARED
    ${primitiv_cuda_LIBRARY_HDRS}
    ${primitiv_cuda_LIBRARY_SRCS}
    OPTIONS -std=c++11
  )
endif()
