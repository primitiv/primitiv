# Build rules of devices.

add_subdirectory(naive)

if(PRIMITIV_USE_EIGEN)
  add_subdirectory(eigen)
endif()

if(PRIMITIV_USE_CUDA)
  add_subdirectory(cuda)
endif()

if(PRIMITIV_USE_OPENCL)
  add_subdirectory(opencl)
endif()

set(primitiv_c_all_OBJS ${primitiv_c_all_OBJS} PARENT_SCOPE)
