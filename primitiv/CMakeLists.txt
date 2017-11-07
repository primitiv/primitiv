# Build rules of the primitiv core library.

# Builds Protobuf messages.
protobuf_generate_cpp(primitiv_proto_SRCS primitiv_proto_HDRS messages.proto)

# Core headers.
set(primitiv_base_HDRS
  ${primitiv_proto_HDRS}
  device.h
  error.h
  function.h
  function_impl.h
  graph.h
  initializer.h
  initializer_impl.h
  mixins.h
  model.h
  naive_device.h
  operators.h
  parameter.h
  primitiv.h
  primitiv_cuda.h
  shape.h
  shape_ops.h
  string_utils.h
  tensor.h
  trainer.h
  trainer_impl.h
  type_traits.h
)

# Core sources.
set(primitiv_base_SRCS
  ${primitiv_proto_SRCS}
  device.cc
  function_impl.cc
  graph.cc
  initializer_impl.cc
  model.cc
  naive_device.cc
  node_ops.cc
  parameter.cc
  shape.cc
  shape_ops.cc
  tensor.cc
  tensor_ops.cc
  trainer.cc
  trainer_impl.cc
)

# Builds core library.
add_library(primitiv_base OBJECT ${primitiv_base_HDRS} ${primitiv_base_SRCS})

set(primitiv_OBJS $<TARGET_OBJECTS:primitiv_base>)
set(primitiv_DEPS ${PROTOBUF_LIBRARIES})
set(primitiv_HDRS ${primitiv_base_HDRS})

# Build rules of the CUDA backend.
if(PRIMITIV_USE_CUDA)
  set(primitiv_cuda_HDRS
    cuda_device.h
    cuda_memory_pool.h
    cuda_utils.h)
  set(primitiv_cuda_SRCS
    cuda_device.cu
    cuda_memory_pool.cu
    cuda_utils.cu)

  list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -O3 -Xcompiler -fPIC")

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

  cuda_compile(primitiv_cuda_OBJS
    ${primitiv_base_HDRS}
    ${primitiv_cuda_HDRS}
    ${primitiv_cuda_SRCS})
  list(APPEND primitiv_OBJS ${primitiv_cuda_OBJS})
  list(APPEND primitiv_DEPS
    ${CUDA_LIBRARIES}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_curand_LIBRARY})
  list(APPEND primitiv_HDRS ${primitiv_cuda_HDRS})
endif()

# Builds the integrated binary.
if(PRIMITIV_BUILD_STATIC_LIBRARY)
  add_library(primitiv STATIC ${primitiv_OBJS})
else()
  add_library(primitiv SHARED ${primitiv_OBJS})
endif()
target_link_libraries(primitiv ${primitiv_DEPS})

# Installs built files.
install(TARGETS primitiv DESTINATION lib)
install(FILES ${primitiv_HDRS} DESTINATION include/primitiv)
