#ifndef PRIMITIV_CORE_MIXINS_NONCOPYABLE_H_
#define PRIMITIV_CORE_MIXINS_NONCOPYABLE_H_

namespace primitiv {
namespace mixins {

/**
 * Mix-in class to prohibit copying.
 */
template<typename T>
class Noncopyable {
private:
  Noncopyable(const Noncopyable &) = delete;
  Noncopyable &operator=(const Noncopyable &) = delete;
protected:
  Noncopyable() = default;
  ~Noncopyable() = default;
};

}  // namespace mixins
}  // namespace primitiv

#endif  // PRIMITIV_CORE_MIXINS_NONCOPYABLE_H_
