# CMake script to find cuDNN library.
#
# Once done, this script will define:
#
#   CUDNN_FOUND - Whether or not the system has the cuDNN library.
#   CUDNN_INCLUDE_DIR - cuDNN include directory.
#   CUDNN_LIBRARIES - cuDNN library files.
#
#   CUDNN_VERSION - Header version of the installed library.
#   CUDNN_VERSION_OK - `ON` if the installed header satisfies version
#                         requirement, `OFF` otherwise.
#
# This script reads hints about search locations from the following
# environment variables:
#
#   CUDNN_ROOT
#   CUDNN_ROOT_DIR
#
# Author: Yusuke Oda (odashi) <yus.takara@gmail.com>
# Created: 2018/01/10
# Updated: 2018/01/10
#
# This script is provided as a part of the "primitiv" core library.
# Redistributing and using this script is allowed under
# the Apache License Version 2.


#
# Finds cuDNN core library
#

find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h
  HINTS
  ${CUDA_TOOLKIT_ROOT_DIR}
  ${CUDNN_ROOT}
  ${CUDNN_ROOT_DIR}
  ENV CUDNN_ROOT
  ENV CUDNN_ROOT_DIR
  PATH_SUFFIXES cuda/include include
)

find_library(CUDNN_LIBRARIES NAMES cudnn
  HINTS
  ${CUDA_TOOLKIT_ROOT_DIR}
  ${CUDNN_ROOT}
  ${CUDNN_ROOT_DIR}
  ENV CUDNN_ROOT
  ENV CUDNN_ROOT_DIR
  PATH_SUFFIXES cuda/lib64 cuda/lib cuda/x64 lib64 lib
)

if (CUDNN_INCLUDE_DIR AND CUDNN_LIBRARIES)
  message("-- Found cuDNN: " ${CUDNN_LIBRARIES})
  #message("-- cuDNN include dir: " ${CUDNN_INCLUDE_DIR})
  #message("-- cuDNN libraries: " ${CUDNN_LIBRARIES})
  set(CUDNN_FOUND ON)
else()
  message(SEND_ERROR "cuDNN is not installed.")
  set(CUDNN_FOUND OFF)
endif()


#
# Checks version
#

if(NOT CuDNN_FIND_VERSION)
  set(CuDNN_FIND_VERSION "${CuDNN_FIND_VERSION_MAJOR}.${CuDNN_FIND_VERSION_MINOR}.${CuDNN_FIND_VERSION_PATCH}")
endif()

file(READ
  "${CUDNN_INCLUDE_DIR}/cudnn.h"
  _cudnn_version_header
)

string(REGEX MATCH
  "#[ \t]*define[ \t]+CUDNN_MAJOR[ \t]+([0-9]+)"
  _cudnn_version_match "${_cudnn_version_header}"
)
set(CUDNN_VERSION_MAJOR "${CMAKE_MATCH_1}")

string(REGEX MATCH
  "#[ \t]*define[ \t]+CUDNN_MINOR[ \t]+([0-9]+)"
  _cudnn_version_match "${_cudnn_version_header}"
)
set(CUDNN_VERSION_MINOR "${CMAKE_MATCH_1}")

string(REGEX MATCH
  "#[ \t]*define[ \t]+CUDNN_PATCHLEVEL[ \t]+([0-9]+)"
  _cudnn_version_match "${_cudnn_version_header}"
)
set(CUDNN_VERSION_PATCH "${CMAKE_MATCH_1}")

set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")

#message("cuDNN version requested: ${CuDNN_FIND_VERSION}")
#message("cuDNN version found: ${CUDNN_VERSION}")

if(${CUDNN_VERSION} VERSION_LESS ${CuDNN_FIND_VERSION})
  message(SEND_ERROR
    "cuDNN version ${CUDNN_VERSION} found in ${CUDNN_INCLUDE_DIR}, "
    "but at least version ${CuDNN_FIND_VERSION} is required."
  )
  set(CUDNN_VERSION_OK OFF)
else()
  set(CUDNN_VERSION_OK ON)
endif()
