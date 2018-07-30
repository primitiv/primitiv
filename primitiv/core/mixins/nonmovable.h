#ifndef PRIMITIV_CORE_MIXINS_NONMOVABLE_H_
#define PRIMITIV_CORE_MIXINS_NONMOVABLE_H_

namespace primitiv {
namespace mixins {

/**
 * Mix-in class to prohibit moving and copying.
 */
template<typename T>
class Nonmovable {
private:
  Nonmovable(const Nonmovable &) = delete;
  Nonmovable(Nonmovable &&) = delete;
  Nonmovable &operator=(const Nonmovable &) = delete;
  Nonmovable &operator=(Nonmovable &&) = delete;
protected:
  Nonmovable() = default;
  ~Nonmovable() = default;
};

}  // namespace mixins
}  // namespace primitiv

#endif  // PRIMITIV_CORE_MIXINS_NONMOVABLE_H_
