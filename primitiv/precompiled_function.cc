#include <primitiv/config.h>

#include <primitiv/precompiled_function.h>

namespace primitiv {

PrecompiledFunction::PrecompiledFunction(const std::string &path)
: lib_(path)
, num_args_fp_(lib_.get_symbol<std::uint32_t(void)>("num_arguments"))
, num_rets_fp_(lib_.get_symbol<std::uint32_t(void)>("num_returns"))
, fwd_shp_fp_(lib_.get_symbol<
    void(const Shape * const * const, Shape * const * const)>("forward_shape"))
, fwd_fp_(lib_.get_symbol<
    void(const Tensor * const * const, Tensor * const * const)>("forward"))
, bwd_fp_(lib_.get_symbol<
    void(
      const Tensor * const * const, const Tensor * const * const,
      const Tensor * const * const, Tensor * const * const)>("backward"))
{}

}  // namespace primitiv
