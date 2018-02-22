#include <primitiv/config.h>

#include <cstdlib>
#include <iostream>

#include <primitiv/dynamic_library.h>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <dlfcn.h>
#else
#error Unsupported system.
#endif

namespace primitiv {

DynamicLibrary::DynamicLibrary(const std::string &path)
: path_(path) {
  ::dlerror();
  handle_ = ::dlopen(path_.c_str(), RTLD_NOW);
  const char *msg = ::dlerror();
  if (msg != NULL) {
    THROW_ERROR(
        "::dlopen() failed. path: '"
        << path_ << "', message: '" << msg << "'");
  }
}

DynamicLibrary::~DynamicLibrary() {
  ::dlerror();
  if (::dlclose(handle_) != 0) {
    const char *msg = ::dlerror();
    const std::string msg_str = msg ? msg : "NULL";
    std::cerr
      << "dlclose() failed. path: '" << path_
      << "', message: '" << msg << "'" << std::endl;
    std::abort();
  }
}

void *DynamicLibrary::get_symbol(const std::string &symbol) {
  ::dlerror();
  void *ptr = ::dlsym(handle_, symbol.c_str());
  const char *msg = ::dlerror();
  if (msg != NULL) {
    THROW_ERROR(
        "::dlsym() failed. path: '"
        << path_ << "', symbol: '" << symbol << "', message: '" << msg << "'");
  }
  return ptr;
}

}  // namespace primitiv
