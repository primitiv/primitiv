#ifndef PRIMITIV_CORE_SHAPE_H_
#define PRIMITIV_CORE_SHAPE_H_

#include <array>
#include <cstdint>
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
 *   Shape({n})      == Shape({n, 1, 1, ...}, 1): column vector
 *   Shape({n, m})   == Shape({n, m, 1, ...}, 1): matrix
 *   Shape({...}, k): k-parallelized data (mini-batch)
 */
class Shape {
public:
  static const std::uint32_t MAX_DEPTH = 8;

  Shape(const Shape &) = default;
  Shape(Shape &&) = default;
  Shape &operator=(const Shape &) = default;
  Shape &operator=(Shape &&);

  /**
   * Creates a new scalar Shape object.
   */
  Shape() : depth_(0), batch_(1), volume_(1) {}

  /**
   * Creates a new Shape object.
   * @param dims List of the dimension sizes.
   * @param batch Batch size.
   */
  Shape(std::initializer_list<std::uint32_t> dims, std::uint32_t batch = 1);

  /**
   * Creates a new Shape object.
   * @param dims List of the dimension sizes.
   * @param batch Batch size.
   */
  Shape(const std::vector<std::uint32_t> &dims, std::uint32_t batch = 1);

  /**
   * Returns the size of the i-th dimension.
   * @param i Dimension number to check.
   * @return Size of the i-th dimension.
   */
  std::uint32_t operator[](std::uint32_t i) const { return i < depth_ ? dims_[i] : 1; }

  /**
   * Returns the dimension array.
   * @return Copy of the dimension array.
   */
  const std::vector<std::uint32_t> dims() const {
    return std::vector<std::uint32_t>(&dims_[0], &dims_[depth_]);
  }

  /**
   * Returns the depth (length of non-1 dimensions) of the shape.
   * @return The depth of the shape.
   */
  std::uint32_t depth() const { return depth_; }

  /**
   * Returns the batch size.
   * @return Batch size.
   */
  std::uint32_t batch() const { return batch_; }

  /**
   * Returns the number of elements in each sample.
   * This value is equal to the product of all dimensions.
   * @return Number of elements.
   */
  std::uint32_t volume() const { return volume_; }

  /**
   * Returns the number of elements in 1 to specified dim.
   * @param dim Upper bound of the dimension.
   * @return `dims[0] * dims[1] * ... * dims[dim-1]`
   */
  std::uint32_t lower_volume(std::uint32_t dim) const {
    std::uint32_t ret = 1, lim = std::min(dim, depth_);
    for (std::uint32_t i = 0; i < lim; ++i) ret *= dims_[i];
    return ret;
  }

  /**
   * Returns the number of elements in all samples of the mini-batch.
   * This value is equal to `batch() * volume()`.
   * @return Number of elements.
   */
  std::uint32_t size() const { return batch_ * volume_; }

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
    return has_same_dims(rhs) && batch_ == rhs.batch_;
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
  bool has_batch() const { return batch_ > 1; }

  /**
   * Checks whether two batch size is compatible (broadcastable) or not.
   * @param rhs Shape object to compare.
   * @return true if both batch size is compatible, false otherwise.
   */
  bool has_compatible_batch(const Shape &rhs) const {
    return batch_ == rhs.batch_ || batch_ == 1 || rhs.batch_ == 1;
  }

  /**
   * Checks whether the shape is a scalar or not.
   * @return true if the shape is a scalar, false otherwise.
   */
  bool is_scalar() const { return depth() == 0; }

  /**
   * Checks whether the shape is a column vector or not.
   * @return true if the shape is a column vector, false otherwise.
   */
  bool is_column_vector() const { return depth() <= 1; }

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
  bool has_same_dims(const Shape &rhs) const {
    bool ok = true;
    for (std::uint32_t i = 0; i < depth_; ++i) ok = ok && dims_[i] == rhs.dims_[i];
    return ok && depth_ == rhs.depth_;
  }

  /**
   * Checks whether two shapes have same dimensions without an axis.
   * (LOO: leave one out)
   * @param rhs Shape object to compare.
   * @param dim Dimension to be ignored.
   * @return true if both shape have same dimensions regardless the dimension
   *         `dim`, false otherwise.
   */
  bool has_same_loo_dims(const Shape &rhs, std::uint32_t dim) const;

  /**
   * Creates a new shape which have one different dimension.
   * @param dim Dimension to be changed.
   * @param m New size of the dimension `dim`.
   * @return New shape.
   */
  Shape resize_dim(std::uint32_t dim, std::uint32_t m) const;

  /**
   * Creates a new shape which have specified batch size.
   * @param batch New batch size.
   * @return New shape.
   */
  Shape resize_batch(std::uint32_t batch) const;

  /**
   * Directly updates a specified dimension.
   * @param dim Dimension to be updated.
   * @param m New size of the dimension `dim`.
   */
  void update_dim(std::uint32_t dim, std::uint32_t m);

  /**
   * Directly updates the batch size.
   * @param batch New batch size.
   */
  void update_batch(std::uint32_t batch);

private:
  std::array<std::uint32_t, MAX_DEPTH> dims_;
  std::uint32_t depth_;
  std::uint32_t batch_;
  std::uint32_t volume_;
};

}  // namespace primitiv

#endif  // PRIMITIV_CORE_SHAPE_H_
