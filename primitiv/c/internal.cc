/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <config.h>

#include <primitiv/c/internal.h>

namespace primitiv_c {

namespace internal {

static thread_local ErrorHandler error_handler;

ErrorHandler &ErrorHandler::get_instance() {
  return error_handler;
}

}  // namespace internal

}  // namespace primitiv_c
