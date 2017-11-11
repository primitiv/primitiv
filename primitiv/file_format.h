#ifndef PRIMITIV_FILE_FORMAT_H_
#define PRIMITIV_FILE_FORMAT_H_

#include <cstdint>
#include <primitiv/error.h>

namespace primitiv {

class FileFormat {
public:
  static const std::uint32_t CURRENT_VERSION = 0x00000000;
  enum class DataType : std::uint32_t {
    SHAPE     = 0x0,
    TENSOR    = 0x100,
    PARAMETER = 0x200,
    MODEL     = 0x300,
    TRAINER   = 0x400,
  };

  static void check_version(std::uint32_t version) {
    if (version != CURRENT_VERSION) {
      THROW_ERROR(
          "File version mismatched. "
          "required: " << std::hex << CURRENT_VERSION
          << ", observed: " << std::hex << version);
    }
  }
};

}  // namespace primitiv

#endif  // PRIMITIV_FILE_FORMAT_H_
