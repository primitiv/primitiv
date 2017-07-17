# test definitions

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${GTEST_INCLUDE_DIRS}
)

function(primitiv_test name)
  add_executable(${name}_test test_utils.h ${name}_test.cc)
  target_link_libraries(${name}_test primitiv ${GTEST_BOTH_LIBRARIES} pthread)
  add_test(
    NAME ${name}_test
    COMMAND ${name}_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endfunction()

primitiv_test(cpu_device)
primitiv_test(device)
primitiv_test(function_impl)
primitiv_test(graph)
primitiv_test(initializer_impl)
primitiv_test(parameter)
primitiv_test(shape)
primitiv_test(shape_ops)
primitiv_test(tensor)
primitiv_test(tensor_backward)
primitiv_test(tensor_ops)
primitiv_test(trainer)
primitiv_test(trainer_impl)

if(PRIMITIV_USE_CUDA)
  primitiv_test(cuda_device)
  primitiv_test(cuda_memory_pool)
endif()
