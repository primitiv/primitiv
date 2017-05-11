#ifndef PRIMITIV_SHAPE_H_
#define PRIMITIV_SHAPE_H_

#include <initializer_list>
#include <string>
#include <vector>

namespace primitiv {

/**
 * Data structure to represent the shape of the node.
 *
 * Examples:
 *   Shape({})       == Shape({1, 1, 1, ...}, 1): scalar
 *   Shape({n})      == Shape({n, 1, 1, ...}, 1): row vector
 *   Shape({n, m})   == Shape({n, m, 1, ...}, 1): matrix
 *   Shape({...}, k): k-parallelized data (mini-batch)
 */
class Shape {
public:
  Shape() = delete;
  Shape(const Shape &) = default;
  Shape(Shape &&) = default;
  Shape &operator=(const Shape &) = default;
  Shape &operator=(Shape && ) = default;
  ~Shape() = default;

  /**
   * Creates a new Shape object.
   * @param dims Integer list to represent the dimension.
   * @param k Batch size.
   */
  Shape(const std::initializer_list<unsigned> dims, const unsigned k = 1);

  /**
   * Returns the size of the i-th dimension.
   * @param i Dimension number to check.
   * @return Size of the i-th dimension.
   */
  inline unsigned dim(const unsigned i) const {
    return i < dims_.size() ? dims_[i] : 1;
  }

  /**
   * Returns the list of dimension sizes.
   * @return List of the dimension sizes.
   */
  inline const std::vector<unsigned> dims() const { return dims_; }

  /**
   * Returns the batch size.
   * @return Batch size.
   */
  inline unsigned batch_size() const { return k_; }

  /**
   * Returns the number of actual data in the node.
   * This value is equal to batch_size() * dim(0) * dim(1) * ...
   * @return Number of actual data in the node.
   */
  inline unsigned size() const {
    unsigned s = k_;
    for (const unsigned d : dims_) s *= d;
    return s;
  }

  /**
   * Returns a string representation of the shape.
   * The format is: "[n,m,...]xk"
   * @return Encoded string.
   */
  std::string to_string() const;

  /**
   * Compare this and another Shape object are same.
   * @param rhs target Shape object to compare.
   * @return true if this and rhs are same, false otherwise.
   */
  inline bool operator==(const Shape &rhs) const {
    return dims_ == rhs.dims_ && k_ == rhs.k_;
  }

  /**
   * Compare this and another Shape object are not same.
   * @param rhs target Shape object to compare.
   * @return true if this and rhs are not same, false otherwise.
   */
  inline bool operator!=(const Shape &rhs) const { return !operator==(rhs); }

private:
  std::vector<unsigned> dims_;
  unsigned k_;

  /**
   * Check internal values and adjust them.
   */
  void adjust();
};

}  // namespace primitiv

#endif  // PRIMITIV_SHAPE_H_
