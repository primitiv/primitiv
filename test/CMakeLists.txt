# test definitions

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/submodules/googletest/googletest/include
)

function(primitiv_test name)
  add_executable(${name}_test test_utils.h ${name}_test.cc)
  target_link_libraries(${name}_test primitiv gtest gtest_main)
  add_test(
    NAME ${name}_test
    COMMAND ${name}_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endfunction()

primitiv_test(cpu_device)
primitiv_test(function_impl)
primitiv_test(graph)
primitiv_test(initializer_impl)
primitiv_test(parameter)
primitiv_test(sgd_trainer)
primitiv_test(shape)
primitiv_test(shape_ops)
primitiv_test(tensor)
primitiv_test(tensor_ops)

if(USE_CUDA)
  primitiv_test(cuda_device)
  target_link_libraries(tensor_test primitiv_cuda)
  target_link_libraries(tensor_ops_test primitiv_cuda)
  target_link_libraries(cuda_device_test primitiv_cuda)
endif()
