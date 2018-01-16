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
#   PRIMITIV_VERSION - Header version of the installed library.
#   PRIMITIV_VERSION_OK - `ON` if the installed header satisfies version
#                         requirement, `OFF` otherwise.
#
# This script reads hints about search locations from the following
# environment variables:
#
#   PRIMITIV_ROOT
#   PRIMITIV_ROOT_DIR
#
# Author: Yusuke Oda (odashi) <yus.takara@gmail.com>
# Created: 2017/08/13
# Updated: 2017/12/16
#
# This script is provided as a part of the "primitiv" core library.
# Redistributing and using this script is allowed under
# the Apache License Version 2.


#
# Finds primitiv core library
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
  message("-- Found primitiv: " ${PRIMITIV_LIBRARIES})
  #message("-- primitiv include dir: " ${PRIMITIV_INCLUDE_DIR})
  #message("-- primitiv libraries: " ${PRIMITIV_LIBRARIES})
  set(PRIMITIV_FOUND ON)
else()
  message(SEND_ERROR "primitiv is not installed.")
  set(PRIMITIV_FOUND OFF)
endif()


#
# Finds primitiv C API
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
  message("-- Found primitiv C API: " ${PRIMITIV_C_LIBRARIES})
  #message("-- primitiv C API include dir: " ${PRIMITIV_C_INCLUDE_DIR})
  #message("-- primitiv C API libraries: " ${PRIMITIV_C_LIBRARIES})
  set(PRIMITIV_C_FOUND ON)
else()
  message(STATUS "primitiv C API is not installed.")
  set(PRIMITIV_C_FOUND OFF)
endif()


#
# Checks version
#

if(NOT Primitiv_FIND_VERSION)
  set(Primitiv_FIND_VERSION "${Primitiv_FIND_VERSION_MAJOR}.${Primitiv_FIND_VERSION_MINOR}.${Primitiv_FIND_VERSION_PATCH}")
endif()

file(READ
  "${PRIMITIV_INCLUDE_DIR}/primitiv/version.h"
  _primitiv_version_header
)

string(REGEX MATCH
  "#[ \t]*define[ \t]+PRIMITIV_VERSION_MAJOR[ \t]+([0-9]+)"
  _primitiv_version_match "${_primitiv_version_header}"
)
set(PRIMITIV_VERSION_MAJOR "${CMAKE_MATCH_1}")

string(REGEX MATCH
  "#[ \t]*define[ \t]+PRIMITIV_VERSION_MINOR[ \t]+([0-9]+)"
  _primitiv_version_match "${_primitiv_version_header}"
)
set(PRIMITIV_VERSION_MINOR "${CMAKE_MATCH_1}")

string(REGEX MATCH
  "#[ \t]*define[ \t]+PRIMITIV_VERSION_PATCH[ \t]+([0-9]+)"
  _primitiv_version_match "${_primitiv_version_header}"
)
set(PRIMITIV_VERSION_PATCH "${CMAKE_MATCH_1}")

set(PRIMITIV_VERSION "${PRIMITIV_VERSION_MAJOR}.${PRIMITIV_VERSION_MINOR}.${PRIMITIV_VERSION_PATCH}")

#message("primitiv version requested: ${Primitiv_FIND_VERSION}")
#message("primitiv version found: ${PRIMITIV_VERSION}")

if(${PRIMITIV_VERSION} VERSION_LESS ${Primitiv_FIND_VERSION})
  message(SEND_ERROR
    "primitiv version ${PRIMITIV_VERSION} found in ${PRIMITIV_INCLUDE_DIR}, "
    "but at least version ${Primitiv_FIND_VERSION} is required."
  )
  set(PRIMITIV_VERSION_OK OFF)
else()
  set(PRIMITIV_VERSION_OK ON)
endif()
