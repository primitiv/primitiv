set(primitiv_c_HDRS
  api.h
  define.h
  device.h
  graph.h
  initializer.h
  internal.h
  model.h
  naive_device.h
  optimizer.h
  parameter.h
  shape.h
  tensor.h
)
set(primitiv_c_SRCS
  device.cc
  graph.cc
  initializer.cc
  model.cc
  naive_device.cc
  optimizer.cc
  parameter.cc
  shape.cc
  tensor.cc)
install(FILES ${primitiv_c_HDRS} DESTINATION include/primitiv_c)

add_library(primitiv_c_OBJS OBJECT
    ${primitiv_c_HDRS}
    ${primitiv_c_SRCS})
set(primitiv_c_all_OBJS $<TARGET_OBJECTS:primitiv_c_OBJS>)

if(PRIMITIV_USE_CUDA)
  set(primitiv_c_cuda_HDRS
    api_cuda.h
    cuda_device.h)
  set(primitiv_c_cuda_SRCS
    cuda_device.cc)
  install(FILES ${primitiv_c_cuda_HDRS} DESTINATION include/primitiv_c)

  add_library(primitiv_c_cuda_OBJS OBJECT
    ${primitiv_c_cuda_HDRS}
    ${primitiv_c_cuda_SRCS})
  list(APPEND primitiv_c_all_OBJS ${primitiv_c_cuda_OBJS})
endif()

if(PRIMITIV_BUILD_STATIC_LIBRARY)
  add_library(primitiv_c STATIC ${primitiv_c_all_OBJS})
else()
  add_library(primitiv_c SHARED ${primitiv_c_all_OBJS})
endif()
target_link_libraries(primitiv_c primitiv)

install(TARGETS primitiv_c DESTINATION lib)