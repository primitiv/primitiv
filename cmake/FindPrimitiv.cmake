# CMake script to find primitiv library.
#
# Once done, this script will define:
#
#   PRIMITIV_FOUND - Whether or not the system has the primitiv library.
#   PRIMITIV_INCLUDE_DIR - primitiv include directory.
#   PRIMITIV_LIBRARIES - primitiv library files.
#
#   PRIMITIV_C_FOUND - Whether or not the system has the primitiv C API.
#   PRIMITIV_C_INCLUDE_DIR - primitiv C API include directory.
#   PRIMITIV_C_LIBRARIES - primitiv C API library files.
#
# This script reads hints about search locations from the following
# environment variables:
#
#   PRIMITIV_ROOT
#   PRIMITIV_ROOT_DIR
#
# Author: Yusuke Oda (odashi) <yus.takara@gmail.com>
# Created: 2017/08/13
# Updated: 2017/12/15
#
# This script is provided as a part of the primitiv core library.
# Redistributing and using this script is allowed according to
# the Apache License Version 2.


#
# Finding primitiv core library
#

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


#
# Finding primitiv C API
#

find_path(PRIMITIV_C_INCLUDE_DIR NAMES primitiv/c/api.h
  HINTS
  ENV PRIMITIV_ROOT
  ENV PRIMITIV_ROOT_DIR
  PATHS
  ${CMAKE_INSTALL_PREFIX}/include
)

find_library(PRIMITIV_C_LIBRARIES NAMES primitiv_c
  HINTS
  ENV PRIMITIV_ROOT
  ENV PRIMITIV_ROOT_DIR
  PATHS
  ${CMAKE_INSTALL_PREFIX}/lib
)

if (PRIMITIV_C_INCLUDE_DIR AND PRIMITIV_C_LIBRARIES)
  message("-- Found Primitiv C API: " ${PRIMITIV_C_LIBRARIES})
  #message("-- Primitiv C API include dir: " ${PRIMITIV_C_INCLUDE_DIR})
  #message("-- Primitiv C API libraries: " ${PRIMITIV_C_LIBRARIES})
  set(PRIMITIV_C_FOUND ON)
else()
  message(STATUS "Primitiv C API is not installed.")
  set(PRIMITIV_C_FOUND OFF)
endif()

