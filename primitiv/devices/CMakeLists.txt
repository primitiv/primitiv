# Build rules of devices.

add_subdirectory(naive)

if(PRIMITIV_USE_EIGEN)
  add_subdirectory(eigen)
endif()

if(PRIMITIV_USE_CUDA)
  add_subdirectory(cuda)
  add_subdirectory(cuda16)
endif()

if(PRIMITIV_USE_OPENCL)
  add_subdirectory(opencl)
endif()

set(primitiv_all_OBJS ${primitiv_all_OBJS} PARENT_SCOPE)
set(primitiv_all_DEPS ${primitiv_all_DEPS} PARENT_SCOPE)
