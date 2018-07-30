#ifndef PRIMITIV_CORE_MIXINS_IDENTIFIABLE_H_
#define PRIMITIV_CORE_MIXINS_IDENTIFIABLE_H_

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include <primitiv/core/error.h>
#include <primitiv/core/mixins/nonmovable.h>

namespace primitiv {
namespace mixins {

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
    if (it == objects_.end()) PRIMITIV_THROW_ERROR("Invalid object ID: " << id);
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

}  // namespace mixins
}  // namespace primitiv

#endif  // PRIMITIV_CORE_MIXINS_IDENTIFIABLE_H_
