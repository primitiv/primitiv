#ifndef PRIMITIV_MIXINS_H_
#define PRIMITIV_MIXINS_H_

#include <primitiv/error.h>

namespace primitiv {
namespace mixins {

/**
 * Mix-in class to provide default value setter/getter.
 */
template<typename T>
class DefaultSettable {
  DefaultSettable(const DefaultSettable &) = delete;
  DefaultSettable(DefaultSettable &&) = delete;
  DefaultSettable &operator=(const DefaultSettable &) = delete;
  DefaultSettable &operator=(DefaultSettable &&) = delete;

  /**
   * Pointer of current default object.
   */
  static T *default_obj_;

public:
  DefaultSettable() = default;

  ~DefaultSettable() {
    // If the current default object is this, unregister it.
    if (default_obj_ == static_cast<T *>(this)) {
      default_obj_ = nullptr;
    }
  }

  /**
   * Retrieves the current default object.
   * @return Reference of the current default object.
   * @throw primitiv::Error Default object is null.
   */
  static T &get_default() {
    if (!default_obj_) THROW_ERROR("Default object is null.");
    return *default_obj_;
  }

  /**
   * Specifies a new default object.
   * @param obj Reference of the new default object.
   */
  static void set_default(T &obj) {
    default_obj_ = &obj;
  }
};

template<typename T>
T *DefaultSettable<T>::default_obj_ = nullptr;

}  // namespace mixins
}  // namespace primitiv

#endif  // PRIMITIV_MIXINS_H_
