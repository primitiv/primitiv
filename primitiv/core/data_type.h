#ifndef PRIMITIV_CORE_DATA_TYPE_H_
#define PRIMITIV_CORE_DATA_TYPE_H_

#include <cstdint>

namespace primitiv {

// Enum for data types in the Tensor.
enum class DataType : std::uint32_t {
  NONE = 0x00000000,
  BOOL = 0x00000001,
  // Reserved: 0x00000002 -- 0x000000ff
  
  INT8 = 0x00000100,
  INT16 = 0x00000101,
  INT32 = 0x00000102,
  INT64 = 0x00000103,
  // Reserved: 0x00000104 -- 0x000001ff
  
  UINT8 = 0x00000200,
  UINT16 = 0x00000201,
  UINT32 = 0x00000202,
  UINT64 = 0x00000203,
  // Reserved: 0x00000204 -- 0x000002ff
  
  FLOAT8 = 0x00000300,
  FLOAT16 = 0x00000301,
  FLOAT32 = 0x00000302,
  FLOAT64 = 0x00000303,
  // Reserved: 0x00000304 -- 0x000003ff
};

}  // namespace primitiv

#endif  // PRIMITIV_CORE_DATA_TYPE_H_
