#ifndef PRIMITIV_CORE_SPINLOCK_H_
#define PRIMITIV_CORE_SPINLOCK_H_

#include <atomic>
#include <cstdint>
#include <thread>

#include <primitiv/core/mixins/nonmovable.h>

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

/**
 * Spinlock object which with std::mutex like interface, and can lock them
 * recursively by the same thread.
 */
class RecursiveSpinlock : mixins::Nonmovable<RecursiveSpinlock> {
  std::atomic_flag ready_ = ATOMIC_FLAG_INIT;
  std::thread::id locked_thread_id_ = std::thread::id();
  std::uint32_t lock_count_ = 0;

public:
  /**
   * Tries to acquire the privilege.
   * @return true if the privilege has been acquired by this call,
   *         false otherwise.
   */
  bool try_lock() {
    const std::thread::id this_thread_id = std::this_thread::get_id();
    if (lock_count_ == 0) {
      if (ready_.test_and_set(std::memory_order_acquire)) {
        return false;
      }
      locked_thread_id_ = this_thread_id;
      ++lock_count_;
      return true;
    } else if (locked_thread_id_ == this_thread_id) {
      ++lock_count_;
      return true;
    }
    return false;
  }

  /**
   * Blocks until the privilege has been acquired.
   */
  void lock() { while (!try_lock()) /* spin */; }

  /**
   * Releases the privilege.
   */
  void unlock() {
    if (locked_thread_id_ != std::this_thread::get_id()) {
      return;
    }
    if (--lock_count_ == 0) {
      locked_thread_id_ = std::thread::id();
      ready_.clear(std::memory_order_release);
    }
  }
};

}  // namespace primitiv

#endif  // PRIMITIV_CORE_SPINLOCK_H_
