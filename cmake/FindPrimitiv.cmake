# CMake script to find primitiv library.
#
# Once done, this script will define:
#
#   PRIMITIV_FOUND - Whether or not the system has the primitiv library.
#   PRIMITIV_INCLUDE_DIR - primitiv include directory.
#   PRIMITIV_LIBRARIES - primitiv library files.
#
# This script reads hints about search locations from the following
# environment variables:
#
#   PRIMITIV_ROOT
#   PRIMITIV_ROOT_DIR
#
# Author: Yusuke Oda (odashi) <yus.takara@gmail.com>
# Created: 2017/08/13
# Updated: 2017/08/13
#
# This script is provided as a part of the primitiv library.
# Redistributing and using this script is allowed according to
# the Apache License Version 2.

find_path(PRIMITIV_INCLUDE_DIR NAMES primitiv/primitiv.h
  HINTS
  ENV PRIMITIV_ROOT
  ENV PRIMITIV_ROOT_DIR
  PATHS
  ${CMAKE_INSTALL_PREFIX}/include
)

find_library(PRIMITIV_LIBRARIES NAMES primitiv
  HINTS
  ENV PRIMITIV_ROOT
  ENV PRIMITIV_ROOT_DIR
  PATHS
  ${CMAKE_INSTALL_PREFIX}/lib
)

if (PRIMITIV_INCLUDE_DIR AND PRIMITIV_LIBRARIES)
  message("-- Found Primitiv: " ${PRIMITIV_LIBRARIES})
  #message("-- Primitiv include dir: " ${PRIMITIV_INCLUDE_DIR})
  #message("-- Primitiv libraries: " ${PRIMITIV_LIBRARIES})
  set(PRIMITIV_FOUND ON)
else()
  message(STATUS "Primitiv is not installed.")
  set(PRIMITIV_FOUND OFF)
endif()
