# CMake script to find OpenCL C++ headers.
#
# Once done, this script will define:
#
#   CLHPP_FOUND - Whether or not the system has the OpenCL C++ headers.
#   CLHPP_INCLUDE_DIR - The include directory which the script found.
#
# This script reads hints about search locations from the following
# environment variables:
#
#   CLHPP_ROOT
#   CLHPP_ROOT_DIR
#
# Author: Yusuke Oda (odashi) <yus.takara@gmail.com>
# Created: 2017/11/16
#
# This script is provided as a part of the primitiv library.
# Redistributing and using this script is allowed according to
# the Apache License Version 2.

find_path(CLHPP_INCLUDE_DIR NAMES CL/cl2.hpp
  HINTS
  ENV CLHPP_ROOT
  ENV CLHPP_ROOT_DIR
  PATHS
  ${CMAKE_INSTALL_PREFIX}/include
)

if (CLHPP_INCLUDE_DIR)
  message("-- Found OpenCL C++ header v2: " ${CLHPP_INCLUDE_DIR})
  set(CLHPP_FOUND ON)
else()
  message(STATUS "OpenCL C++ header v2 is not installed.")
  set(CLHPP_FOUND OFF)
endif()
