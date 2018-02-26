configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/location.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/location.h
  @ONLY
)

# Custom build rule for DLLs.
# @param name Name of the DLL. This value should be the source file name except
#             the extension (e.g., "foo_lib.cc" -> "foo_lib")
function(primitiv_dll name)
  add_library(${name} SHARED ${name}.cc)
  set_target_properties(${name} PROPERTIES PREFIX "" SUFFIX ".dll")
endfunction()

# Compile options of NVCC.
list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -O3 -Xcompiler -fPIC")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_52,code=sm_52")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_52,code=compute_52")
if(CUDA_VERSION VERSION_EQUAL 8.0 OR CUDA_VERSION VERSION_GREATER 8.0)
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_60,code=sm_60")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=compute_61")
endif()
if(CUDA_VERSION VERSION_EQUAL 9.0 OR CUDA_VERSION VERSION_GREATER 9.0)
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_70,code=sm_70")
  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_70,code=compute_70")
endif()

# Custom build rule for CUDA DLLs.
# @param name Name of the DLL. This value should be the source file name except
#             the extension (e.g., "foo_lib.cc" -> "foo_lib")
function(primitiv_cuda_dll name)
  cuda_add_library(${name} ${name}.cu SHARED)
  set_target_properties(${name} PROPERTIES PREFIX "" SUFFIX ".cudll")
endfunction()

primitiv_dll(basic_test)
primitiv_cuda_dll(precompiled_function)
