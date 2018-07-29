#include <primitiv/config.h>

#include <primitiv/c/internal/internal.h>

namespace primitiv {

namespace c {

namespace internal {

static thread_local ErrorHandler error_handler;

ErrorHandler &ErrorHandler::get_instance() {
  return error_handler;
}

}  // namespace internal

}  // namespace c

}  // namespace primitiv
