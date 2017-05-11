# test definitions
function(primitiv_test name)
  add_executable(${name}_test ${name}_test.cc)
  target_link_libraries(${name}_test primitiv gtest gtest_main)
  add_test(
    NAME ${name}_test
    COMMAND ${name}_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endfunction()

primitiv_test(cpu_device)
primitiv_test(shape)
primitiv_test(tensor)
primitiv_test(tensor_ops)
