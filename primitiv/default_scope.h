#ifndef PRIMITIV_DEFAULT_SCOPE_H_
#define PRIMITIV_DEFAULT_SCOPE_H_

#include <cstdlib>
#include <iostream>
#include <stack>

#include <primitiv/error.h>
#include <primitiv/type_traits.h>

namespace primitiv {

/**
 * Manages the scope of default object.
 */
template<typename T>
class DefaultScope {
  DefaultScope(const DefaultScope &) = delete;
  DefaultScope(DefaultScope &&) = delete;
  DefaultScope &operator=(const DefaultScope &) = delete;
  DefaultScope &operator=(DefaultScope &&) = delete;

  // Type enabler
  using obj_type = typename std::enable_if<
    type_traits::is_scoped<T>::value, T>::type;

  // Entry of the default scope stack.
  struct Entry {
    const DefaultScope *scope;
    T *ptr;
  };

  // Customized stack.
  class Stack : public std::stack<Entry> {
  public:
    ~Stack() {
      if (!Stack::empty()) {
        std::cerr
          << "FATAL ERROR: Default scope stack is not empty!" << std::endl;
        std::cerr << "  #remained scopes: " << size() << std::endl;
        std::abort();
      }
    }
  };

  static Stack stack_;

public:
  /**
   * Enters a scope with the null object, i.e., makes the current scope invalid.
   */
  DefaultScope() {
    stack_.push(Entry {this, nullptr});
  }

  /**
   * Enters a scope with a valid object.
   * @param obj Target object.
   */
  explicit DefaultScope(T &obj) {
    stack_.push(Entry {this, &obj});
  }

  /**
   * Leaves the scope.
   */
  ~DefaultScope() {
    if (stack_.empty()) {
      // Stack size should be equal to the number of DefaultScope<T> objects.
      // So the program should never arrive here.
      std::cerr << "FATAL ERROR: Default scope stack is empty!" << std::endl;
      std::abort();
    }
    const auto top = stack_.top();
    if (top.scope != this) {
      std::cerr
        << "FATAL ERROR: Attempted to delete inordered default scope!"
        << std::endl;
      std::cerr << "  #remained scopes: " << stack_.size() << std::endl;
      std::cerr << "  Current scope: " << top.scope << std::endl;
      std::cerr << "  Scope to be deleted: " << this << std::endl;
      std::abort();
    }
    stack_.pop();
  }

  /**
   * Retrieves the object registered to current scope.
   * @return Registered object.
   */
  static T &get() {
    if (stack_.empty()) THROW_ERROR("No default scope registered.");
    const auto top = stack_.top();
    if (!top.ptr) THROW_ERROR("Default scope is null.");
    return *top.ptr;
  }

  /**
   * Retrieves the depth of the default scope stack.
   * @return Depth of the stack.
   */
  static size_t size() { return stack_.size(); }
};

template<typename T> typename DefaultScope<T>::Stack DefaultScope<T>::stack_;

}  // namespace primitiv

#endif  // PRIMITIV_DEFAULT_SCOPE_H_
