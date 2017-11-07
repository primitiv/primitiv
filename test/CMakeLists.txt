# test definitions

include(ExternalProject)

if(PRIMITIV_GTEST_SOURCE_DIR)
  # Use specific googletest.
  message("-- Import googletest from: ${PRIMITIV_GTEST_SOURCE_DIR}")
  ExternalProject_Add(googletest
    SOURCE_DIR ${PRIMITIV_GTEST_SOURCE_DIR}
    INSTALL_COMMAND ""
    TEST_COMMAND ""
  )
else()
  # Download googletest from GitHub.
  set(gtest_repo https://github.com/google/googletest.git)
  set(gtest_tag release-1.8.0)
  message("-- Import googletest from: ${gtest_repo} (${gtest_tag})")
  ExternalProject_Add(googletest
    GIT_REPOSITORY ${gtest_repo}
    GIT_TAG ${gtest_tag}
    SOURCE_DIR "${CMAKE_BINARY_DIR}/third_party/googletest-src"
    BINARY_DIR "${CMAKE_BINARY_DIR}/third_party/googletest-build"
    INSTALL_COMMAND ""
    TEST_COMMAND ""
  )
endif()

ExternalProject_Get_Property(googletest binary_dir)
add_library(gtest UNKNOWN IMPORTED)
add_library(gtest_main UNKNOWN IMPORTED)
set_property(
  TARGET gtest
  PROPERTY IMPORTED_LOCATION ${binary_dir}/googlemock/gtest/libgtest.a
)
set_property(
  TARGET gtest_main
  PROPERTY IMPORTED_LOCATION ${binary_dir}/googlemock/gtest/libgtest_main.a
)
add_dependencies(gtest googletest)
add_dependencies(gtest_main googletest)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${GTEST_INCLUDE_DIRS}
)

function(primitiv_test name)
  add_executable(${name}_test test_utils.h ${name}_test.cc)
  add_dependencies(${name}_test googletest)
  target_link_libraries(${name}_test primitiv gtest gtest_main pthread)
  add_test(
    NAME ${name}_test
    COMMAND ${name}_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endfunction()

primitiv_test(device)
primitiv_test(function_impl)
primitiv_test(graph)
primitiv_test(initializer_impl)
primitiv_test(mixins)
primitiv_test(naive_device)
primitiv_test(node)
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
