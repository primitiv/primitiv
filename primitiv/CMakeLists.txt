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
  shape.h
  shape_ops.h
  tensor.h
  tensor_ops.h
  trainer.h
  trainer_impl.h
)

set(primitiv_LIBRARY_SRCS
  cpu_device.cc
  device.cc
  function_impl.cc
  graph.cc
  initializer_impl.cc
  node_ops.cc
  parameter.cc
  shape.cc
  shape_ops.cc
  tensor.cc
  tensor_ops.cc
  trainer_impl.cc
)

add_library(primitiv SHARED ${primitiv_LIBRARY_HDRS} ${primitiv_LIBRARY_SRCS})
target_link_libraries(primitiv)

if(USE_CUDA)
  set(primitiv_cuda_LIBRARY_HDRS
    ${primitiv_LIBRARY_HDRS}
    cuda_device.h
    cuda_memory_pool.h
    cuda_utils.h
  )
  set(primitiv_cuda_LIBRARY_SRCS
    cuda_device.cu
    cuda_memory_pool.cu
    cuda_utils.cu
  )
  list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

  # Workaround for some systems.
  #list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES")
  #list(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED")

  # Supported architectures.
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_37,code=sm_37")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_52,code=sm_52")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_52,code=compute_52")

  cuda_add_library(primitiv_cuda SHARED
    ${primitiv_cuda_LIBRARY_HDRS}
    ${primitiv_cuda_LIBRARY_SRCS}
  )
  target_link_libraries(primitiv_cuda
    ${CUDA_cublas_LIBRARY}
    ${CUDA_curand_LIBRARY}
  )
endif()
