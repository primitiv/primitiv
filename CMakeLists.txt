cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(primitiv VERSION 0.1.0 LANGUAGES CXX)

# utility to check if required option is set.
function(my_check_option name)
  if(DEFINED ENV{${name}} AND NOT DEFINED ${name})
    set(${name} "$ENV{${name}}")
  endif()
  if(${name})
    MESSAGE("-- ${name}: ${${name}}")
  else()
    MESSAGE(FATAL_ERROR "Please give the variable `${name}`.")
  endif()
endfunction()

# compiler requirements and settings
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -Wall -Werror")
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}/submodules/googletest/googletest/include
)

link_directories(
)

set(BUILD_SHARED_LIBS ON)

enable_testing()

# sources
add_subdirectory(primitiv)

# tests
add_subdirectory(submodules/googletest)
add_subdirectory(test)
