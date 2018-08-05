#ifndef PRIMITIV_CORE_ERROR_H_
#define PRIMITIV_CORE_ERROR_H_

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace primitiv {

/**
 * A library specific exception.
 */
class Error : public std::exception {
  Error() = delete;

public:
  Error(const std::string &file, std::uint32_t line, const std::string &message)
  : file_(file), line_(line), msg_(message) {
    std::stringstream ss;
    ss << file_ << ": " << line_ << ": " << msg_;
    full_msg_ = ss.str();
  }

  const char *what() const noexcept override { return full_msg_.c_str(); }

private:
  std::string file_;
  std::uint32_t line_;
  std::string msg_;
  std::string full_msg_;
};

/**
 * Not-implemented signal.
 */
class NotImplementedError : public Error {
public:
  NotImplementedError(
      const std::string &file, std::uint32_t line, const std::string &message)
  : Error(file, line, "Not implemented: " + message) {}
};

}  // namespace primitiv

#define PRIMITIV_THROW_ERROR(cmds) { \
  std::stringstream ss; \
  ss << cmds; \
  throw primitiv::Error(__FILE__, __LINE__, ss.str()); \
}

#define PRIMITIV_THROW_NOT_IMPLEMENTED { \
  throw primitiv::NotImplementedError(__FILE__, __LINE__, __func__); \
}

#define PRIMITIV_THROW_NOT_IMPLEMENTED_WITH_MESSAGE(cmds) { \
  std::stringstream ss; \
  ss << cmds; \
  throw primitiv::NotImplementedError(__FILE__, __LINE__, ss.str()); \
}

#endif  // PRIMITIV_CORE_ERROR_H_
