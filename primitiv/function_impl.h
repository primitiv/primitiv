#ifndef PRIMITIV_FUNCTION_IMPL_H_
#define PRIMITIV_FUNCTION_IMPL_H_

#include <primitiv/function.h>

namespace primitiv {
namespace functions {

#define DECL_FUNC(name_) \
  class name_ : public Function { \
  public: \
    Shape forward_shape( \
        const std::vector<const Shape *> &args) const override; \
    Tensor forward(const std::vector<const Tensor *> &args) const override; \
    std::string name() const override { return #name_; } \
  }

/**
 * Function object that behaves data source of the computation graph.
 */
class Input : public primitiv::Function {
public:
  Input() = delete;

  /**
   * Creates an input function.
   * @param shape The shape of the input data.
   */
  Input(const Shape &shape);
  
  Shape forward_shape(const std::vector<const Shape *> &args) const override;
  Tensor forward(const std::vector<const Tensor *> &args) const override;
  std::string name() const override { return "Input"; }

private:
  Shape shape_;
};

//DECL_FUNC(Parameter);

// Arithmetic functions
DECL_FUNC(Add);
DECL_FUNC(AddConst);
DECL_FUNC(Subtract);
DECL_FUNC(SubtractConstL);
DECL_FUNC(SubtractConstR);
DECL_FUNC(Multiply);
DECL_FUNC(MultiplyConst);
DECL_FUNC(Divide);
DECL_FUNC(DivideConstL);
DECL_FUNC(DivideConstR);

#undef DECL_FUNC

}  // namespace functions
}  // namespace primitiv

#endif  // PRIMITIV_FUNCTION_IMPL_H_
