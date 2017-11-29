#ifndef PRIMITIV_SPINLOCK_H_
#define PRIMITIV_SPINLOCK_H_

#include <atomic>

#include <primitiv/mixins.h>

namespace primitiv {

/**
 * Spinlock object which can be used with std::mutex style usage.
 */
class Spinlock : mixins::Nonmovable<Spinlock> {
  std::atomic_flag ready_ = ATOMIC_FLAG_INIT;
public:
  /**
   * Tries to acquire the privilege.
   * @return true if the privilege has been acquired by this call,
   *         false otherwise.
   */
  bool try_lock() { return !ready_.test_and_set(std::memory_order_acquire); }

  /**
   * Blocks until the privilege has been acquired.
   */
  void lock() { while (!try_lock()) /* spin */; }

  /**
   * Releases the privilege.
   */
  void unlock() { ready_.clear(std::memory_order_release); }
};

}  // namespace primitiv

#endif  // PRIMITIV_SPINLOCK_H_
