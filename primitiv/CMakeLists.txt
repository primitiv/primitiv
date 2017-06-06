set(primitiv_LIBRARY_HDRS
  cpu_device.h
  device.h
  error.h
  function.h
  function_impl.h
  graph.h
  initializer.h
  initializer_impl.h
  node.h
  node_ops.h
  parameter.h
  primitiv.h
  primitiv_cuda.h
  sgd_trainer.h
  shape.h
  tensor.h
  tensor_ops.h
  trainer.h
)

set(primitiv_LIBRARY_SRCS
  cpu_device.cc
  device.cc
  function_impl.cc
  graph.cc
  initializer_impl.cc
  parameter.cc
  sgd_trainer.cc
  shape.cc
  tensor.cc
  tensor_ops.cc
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
  target_link_libraries(primitiv_cuda
    ${CUDA_cublas_LIBRARY}
    ${CUDA_curand_LIBRARY}
  )
endif()
