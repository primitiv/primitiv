# CMake script to find CLBlast.
#
# Once done, this script will define:
#
#   CLBLAST_FOUND - Whether or not the system has the CLBlast library.
#   CLBLAST_INCLUDE_DIR - CLBlast include directory.
#   CLBLAST_LIBRARIES - CLBlast library files.
#
#   CLBLAST_VERSION - Header version of the installed library.
#   CLBLAST_VERSION_OK - `ON` if the installed header satisfies version
#                        requirement, `OFF` otherwise.
#
# This script reads hints about search locations from the following
# environment variables:
#
#   CLBLAST_ROOT
#   CLBLAST_ROOT_DIR
#
# Author: Yusuke Oda (odashi) <yus.takara@gmail.com>
#         Koichi Akabe (vbkaisetsu) <vbkaisetsu@gmail.com>
# Created: 2017/03/03
# Updated: 2017/03/03
#
# This script is provided as a part of the primitiv core library.
# Redistributing and using this script is allowed according to
# the Apache License Version 2.


#
# Finds CLBlast
#

find_path(CLBLAST_INCLUDE_DIR NAMES clblast.h
  HINTS
  ${CLBLAST_ROOT}
  ${CLBLAST_ROOT_DIR}
  ENV CLBLAST_ROOT
  ENV CLBLAST_ROOT_DIR
  PATHS
  ${CMAKE_INSTALL_PREFIX}/include
)

find_library(CLBLAST_LIBRARIES NAMES clblast
  HINTS
  ${CLBLAST_ROOT}
  ${CLBLAST_ROOT_DIR}
  ENV CLBLAST_ROOT
  ENV CLBLAST_ROOT_DIR
  PATHS
  ${CMAKE_INSTALL_PREFIX}/lib
)

if (CLBLAST_INCLUDE_DIR AND CLBLAST_LIBRARIES)
  message("-- Found CLBlast: " ${CLBLAST_LIBRARIES})
  #message("-- CLBlast include dir: " ${CLBLAST_INCLUDE_DIR})
  #message("-- CLBlast libraries: " ${CLBLAST_LIBRARIES})
  set(CLBLAST_FOUND ON)
else()
  message(STATUS "CLBlast is not installed.")
  set(CLBLAST_FOUND OFF)
endif()

#
# Checks version
#

if(NOT CLBlast_FIND_VERSION)
  set(CLBlast_FIND_VERSION "${CLBlast_FIND_VERSION_MAJOR}.${CLBlast_FIND_VERSION_MINOR}.${CLBlast_FIND_VERSION_PATCH}")
endif()

get_filename_component(CLBLAST_LIBRARY_DIR ${CLBLAST_LIBRARIES} DIRECTORY)
file(READ
  "${CLBLAST_LIBRARY_DIR}/pkgconfig/clblast.pc"
  _clblast_pkgconfig
)

string(REGEX MATCH
  "Version: ([0-9]+)\.([0-9]+)\.([0-9]+)"
  _clblast_version_match "${_clblast_pkgconfig}"
)
set(CLBLAST_VERSION_MAJOR "${CMAKE_MATCH_1}")
set(CLBLAST_VERSION_MINOR "${CMAKE_MATCH_2}")
set(CLBLAST_VERSION_PATCH "${CMAKE_MATCH_3}")

set(CLBLAST_VERSION "${CLBLAST_VERSION_MAJOR}.${CLBLAST_VERSION_MINOR}.${CLBLAST_VERSION_PATCH}")

#message("CLBlast version requested: ${CLBlast_FIND_VERSION}")
#message("CLBlast version found: ${CLBLAST_VERSION}")

if(${CLBLAST_VERSION} VERSION_LESS ${CLBlast_FIND_VERSION})
  message(STATUS
    "CLBlast version ${CLBLAST_VERSION} found in ${CLBLAST_INCLUDE_DIR}, "
    "but at least version ${CLBlast_FIND_VERSION} is required."
  )
  set(CLBLAST_VERSION_OK OFF)
else()
  set(CLBLAST_VERSION_OK ON)
endif()
