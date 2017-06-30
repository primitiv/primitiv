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
 *   Shape()         == Shape({1, 1, 1, ...}, 1): scalar
 *   Shape({})       == Shape({1, 1, 1, ...}, 1): scalar
 *   Shape({n})      == Shape({n, 1, 1, ...}, 1): row vector
 *   Shape({n, m})   == Shape({n, m, 1, ...}, 1): matrix
 *   Shape({...}, k): k-parallelized data (mini-batch)
 */
class Shape {
public:
  Shape(const Shape &) = default;
  Shape(Shape &&) = default;
  Shape &operator=(const Shape &) = default;
  Shape &operator=(Shape &&);

  /**
   * Creates a new scalar Shape object.
   */
  Shape() : dims_(), k_(1) { adjust(); }

  /**
   * Creates a new Shape object.
   * @param dims List of the dimension sizes.
   * @param k Batch size.
   */
  Shape(std::initializer_list<unsigned> dims, const unsigned k = 1)
    : dims_(dims), k_(k) { adjust(); }

  /**
   * Creates a new Shape object.
   * @param dims List of the dimension sizes.
   * @param k Batch size.
   */
  Shape(const std::vector<unsigned> &dims, const unsigned k = 1)
    : dims_(dims), k_(k) { adjust(); }

  /**
   * Creates a new Shape object.
   * @param dims List of the dimension sizes.
   * @param k Batch size.
   */
  Shape(std::vector<unsigned> &&dims, const unsigned k = 1)
    : dims_(std::move(dims)), k_(k) { adjust(); }

  /**
   * Returns the size of the i-th dimension.
   * @param i Dimension number to check.
   * @return Size of the i-th dimension.
   */
  unsigned operator[](const unsigned i) const {
    return i < depth() ? dims_[i] : 1;
  }

  /**
   * Returns the depth (length of non-1 dimensions) of the shape.
   * @return The depth of the shape.
   */
  unsigned depth() const { return dims_.size(); }

  /**
   * Returns the batch size.
   * @return Batch size.
   */
  unsigned batch() const { return k_; }

  /**
   * Returns the number of elements in each sample.
   * This value is equal to the product of all dimensions.
   * @return Number of elements.
   */
  unsigned volume() const { return volume_; }

  /**
   * Returns the number of elements in 1 to specified dim.
   * @param rank Upper bound of the dimension.
   * @return `dim[0] * dim[1] * ... * dim[dim-1]`
   */
  unsigned lower_volume(unsigned dim) const;

  /**
   * Returns the number of elements in all samples of the mini-batch.
   * This value is equal to `batch() * volume()`.
   * @return Number of elements.
   */
  unsigned size() const { return k_ * volume_; }

  /**
   * Returns a string representation of the shape.
   * The format is: "[n,m,...]xk"
   * @return Encoded string.
   */
  std::string to_string() const;

  /**
   * Compares this and other shape.
   * @param rhs Shape object to compare.
   * @return true if this and rhs are same, false otherwise.
   */
  bool operator==(const Shape &rhs) const {
    return has_same_dims(rhs) && k_ == rhs.k_;
  }

  /**
   * Compares this and other shape.
   * @param rhs Shape object to compare.
   * @return true if this and rhs are not same, false otherwise.
   */
  bool operator!=(const Shape &rhs) const { return !operator==(rhs); }

  /**
   * Checks whether the shape has minibatch or not.
   * @return true if the shape has minibatch, false otherwise.
   */
  bool has_batch() const { return k_ > 1; }

  /**
   * Checks whether two batch size is compatible (broadcastable) or not.
   * @param rhs Shape object to compare.
   * @return true if both batch size is compatible, false otherwise.
   */
  bool has_compatible_batch(const Shape &rhs) const {
    return k_ == rhs.k_ || k_ == 1 || rhs.k_ == 1;
  }

  /**
   * Checks whether the shape is a scalar or not.
   * @return true if the shape is a scalar, false otherwise.
   */
  bool is_scalar() const { return depth() == 0; }

  /**
   * Checks whether the shape is a row vector or not.
   * @return true if the shape is a row vector, false otherwise.
   */
  bool is_row_vector() const { return depth() <= 1; }

  /**
   * Checks whether the shape is a vector or a matrix, or not.
   * @return true if the shape is a vector or a matrix, false otherwise.
   */
  bool is_matrix() const { return depth() <= 2; }

  /**
   * Checks whether two shapes have completely same dimensions.
   * @param rhs Shape object to compare.
   * @return true if both shape have same dimensions, false otherwise.
   */
  bool has_same_dims(const Shape &rhs) const { return dims_ == rhs.dims_; }

  /**
   * Checks whether two shapes have same dimensions without an axis.
   * (LOO: leave one out)
   * @param rhs Shape object to compare.
   * @param dim Dimension to be ignored.
   * @return true if both shape have same dimensions regardless the dimension
   *         `dim`, false otherwise.
   */
  bool has_same_loo_dims(const Shape &rhs, unsigned dim) const;

  /**
   * Creates a new shape which have one different dimension.
   * @param dim Dimension to be changed.
   * @param m New size of the dimension `dim`.
   * @return New shape.
   */
  Shape resize_dim(unsigned dim, unsigned m) const;

  /**
   * Creates a new shape which have specified batch size.
   * @param k New batch size.
   * @return New shape.
   */
  Shape resize_batch(unsigned k) const;

  /**
   * Directly updates a specified dimension.
   * @param dim Dimension to be updated.
   * @param m New size of the dimension `dim`.
   */
  void update_dim(unsigned dim, unsigned m);

  /**
   * Directly updates the batch size.
   * @param k New batch size.
   */
  void update_batch(unsigned k);

private:
  std::vector<unsigned> dims_;
  unsigned k_;
  unsigned volume_;

  /**
   * Check internal values and adjust them.
   */
  void adjust();
};

}  // namespace primitiv

#endif  // PRIMITIV_SHAPE_H_
