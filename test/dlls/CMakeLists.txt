configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/location.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/location.h
  @ONLY
)

add_library(basic_test SHARED basic_test.cc)

set_target_properties(basic_test PROPERTIES PREFIX "" SUFFIX ".dll")
