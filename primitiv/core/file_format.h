#ifndef PRIMITIV_CORE_FILE_FORMAT_H_
#define PRIMITIV_CORE_FILE_FORMAT_H_

#include <cstdint>

#include <primitiv/core/error.h>

namespace primitiv {

class FileFormat {
public:
  class CurrentVersion {
  public:
    static const std::uint32_t MAJOR = 0;
    static const std::uint32_t MINOR = 1;
  };

  enum class DataType : std::uint32_t {
    SHAPE     = 0x0,
    TENSOR    = 0x100,
    PARAMETER = 0x200,
    MODEL     = 0x300,
    OPTIMIZER = 0x400,
  };

  static void assert_version(std::uint32_t major, std::uint32_t minor) {
    if (major != CurrentVersion::MAJOR || minor != CurrentVersion::MINOR) {
      PRIMITIV_THROW_ERROR(
          "File version mismatched. required: "
          << CurrentVersion::MAJOR << "." << CurrentVersion::MINOR
          << ", observed: "
          << major << "." << minor);
    }
  }

  static void assert_datatype(DataType required, std::uint32_t observed) {
    if (observed != static_cast<std::uint32_t>(required)) {
      PRIMITIV_THROW_ERROR(
          "Data type mismatched. required: "
          << std::hex << static_cast<std::uint32_t>(required)
          << ", observed: "
          << observed);
    }
  }
};

}  // namespace primitiv

#endif  // PRIMITIV_CORE_FILE_FORMAT_H_
