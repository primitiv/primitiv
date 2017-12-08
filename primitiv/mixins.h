#ifndef PRIMITIV_MIXINS_H_
#define PRIMITIV_MIXINS_H_

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <primitiv/error.h>

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

/**
 * Mix-in class to allow to bind object IDs.
 */
template<typename T>
class Identifiable : Nonmovable<Identifiable<T>> {
  static std::uint64_t next_id_;
  static std::unordered_map<std::uint64_t, T *> objects_;
  static std::mutex mutex_;
  std::uint64_t id_;

protected:
  Identifiable() {
    const std::lock_guard<std::mutex> lock(mutex_);
    id_ = next_id_++;
    objects_.emplace(id_, static_cast<T *>(this));
  }

  ~Identifiable() {
    const std::lock_guard<std::mutex> lock(mutex_);
    objects_.erase(id_);
  }

public:
  /**
   * Obtains the object which has specified ID.
   * @return Reference of the object.
   * @throw primitiv::Error Object does not exist.
   */
  static T &get_object(std::uint64_t id) {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto it = objects_.find(id);
    if (it == objects_.end()) THROW_ERROR("Invalid object ID: " << id);
    return *it->second;
  }

  /**
   * Obtains the object ID.
   * @return object ID.
   */
  std::uint64_t id() const { return id_; }
};

template<typename T>
std::uint64_t Identifiable<T>::next_id_ = 0;
template<typename T>
std::unordered_map<std::uint64_t, T *> Identifiable<T>::objects_;
template<typename T>
std::mutex Identifiable<T>::mutex_;

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

#endif  // PRIMITIV_MIXINS_H_
