# Build rules of the Eigen backend.

file(GLOB primitiv_eigen_HDRS "*.h")
file(GLOB primitiv_eigen_SRCS "*.cc")
file(GLOB primitiv_eigen_ops_HDRS "ops/*.h")
file(GLOB primitiv_eigen_ops_SRCS "ops/*.cc")
install(
  FILES ${primitiv_eigen_HDRS} DESTINATION include/primitiv/devices/eigen)

set(primitiv_eigen_COMPILE_FLAGS "")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(primitiv_eigen_COMPILE_FLAGS
    "${primitiv_eigen_COMPILE_FLAGS} -march=native")
  # NOTE(odashi):
  # Explicitly disabling `-Wint-in-bool-context` to avoid warnings caused by
  # Eigen v3.3.4 or earlier.
  if(CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 7.0 OR
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    if (EIGEN3_VERSION VERSION_LESS 3.3.5)
      set(primitiv_eigen_COMPILE_FLAGS
        "${primitiv_eigen_COMPILE_FLAGS} -Wno-int-in-bool-context")
    endif()
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(primitiv_eigen_COMPILE_FLAGS
    "${primitiv_eigen_COMPILE_FLAGS} -march=native")
endif()

set_source_files_properties(
  ${primitiv_eigen_ops_SRCS}
  PROPERTIES COMPILE_FLAGS ${primitiv_eigen_COMPILE_FLAGS}
)

add_library(primitiv_eigen_OBJS OBJECT
  ${primitiv_core_HDRS}
  ${primitiv_eigen_HDRS}
  ${primitiv_eigen_SRCS}
  ${primitiv_eigen_ops_HDRS}
  ${primitiv_eigen_ops_SRCS}
)

list(APPEND primitiv_all_OBJS $<TARGET_OBJECTS:primitiv_eigen_OBJS>)

set(primitiv_all_OBJS ${primitiv_all_OBJS} PARENT_SCOPE)
