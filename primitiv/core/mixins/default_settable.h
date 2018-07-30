#ifndef PRIMITIV_CORE_MIXINS_DEFAULT_SETTABLE_H_
#define PRIMITIV_CORE_MIXINS_DEFAULT_SETTABLE_H_

#include <primitiv/core/error.h>

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

protected:
  DefaultSettable() = default;

  ~DefaultSettable() {
    // If the current default object is this, unregister it.
    if (default_obj_ == static_cast<T *>(this)) {
      default_obj_ = nullptr;
    }
  }

public:
  /**
   * Retrieves the current default object.
   * @return Reference of the current default object.
   * @throw primitiv::Error Default object is null.
   */
  static T &get_default() {
    if (!default_obj_) PRIMITIV_THROW_ERROR("Default object is null.");
    return *default_obj_;
  }

  /**
   * Specifies a new default object.
   * @param obj Reference of the new default object.
   */
  static void set_default(T &obj) {
    default_obj_ = &obj;
  }

  /**
   * Obtains the reference of the object pointed by a pointer, or obtains the
   * default object.
   * @param ptr Pointer of an object, or `nullptr`.
   * @return Reference of the object pointed by `ptr`,
   *         or reference of the default object if `ptr` is `nullptr`.
   * @throw primitiv::Error `nullptr` is given by `ptr` although the default
   *                        object is also null.
   */
  static T &get_reference_or_default(T *ptr) {
    return ptr ? *ptr : get_default();
  }
};

template<typename T>
T *DefaultSettable<T>::default_obj_ = nullptr;

}  // namespace mixins
}  // namespace primitiv

#endif  // PRIMITIV_CORE_MIXINS_DEFAULT_SETTABLE_H_
