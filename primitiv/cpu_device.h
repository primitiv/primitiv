#ifndef PRIMITIV_CPU_DEVICE_H_
#define PRIMITIV_CPU_DEVICE_H_

#include <map>
#include <primitiv/device.h>

namespace primitiv {

/**
 * Device class for the usual CPU.
 */
class CPUDevice : public Device {
  CPUDevice(const CPUDevice &) = delete;
  CPUDevice(CPUDevice &&) = delete;
  CPUDevice &operator=(const CPUDevice &) = delete;
  CPUDevice &operator=(CPUDevice &&) = delete;

public:
  CPUDevice() = default;
  ~CPUDevice() override;

  void *allocate(const unsigned size) override;
  void free(void *ptr) override;
  void copy_to_device(
      void *dest, const void *src, const unsigned size) override;
  void copy_to_host(
      void *dest, const void *src, const unsigned size) override;
  unsigned num_blocks() const override { return blocks_.size(); }

  Tensor add(const Tensor &x, const float k) override;
  Tensor add(const Tensor &a, const Tensor &b) override;
  Tensor subtract(const Tensor &x, const float k) override;
  Tensor subtract(const float k, const Tensor &x) override;
  Tensor subtract(const Tensor &a, const Tensor &b) override;
  Tensor multiply(const Tensor &x, const float k) override;
  Tensor multiply(const Tensor &a, const Tensor &b) override;
  Tensor divide(const Tensor &x, const float k) override;
  Tensor divide(const float k, const Tensor &x) override;
  Tensor divide(const Tensor &a, const Tensor &b) override;

  void aug_add(Tensor &x, const float k) override;
  void aug_add(Tensor &a, const Tensor &b) override;
  void aug_subtract(Tensor &x, const float k) override;
  void aug_subtract(Tensor &a, const Tensor &b) override;
  void aug_multiply(Tensor &x, const float k) override;
  void aug_multiply(Tensor &a, const Tensor &b) override;
  void aug_divide(Tensor &x, const float k) override;
  void aug_divide(Tensor &a, const Tensor &b) override;

private:
  std::map<void *, unsigned> blocks_;
};

}  // namespace primitiv

#endif  // PRIMITIV_CPU_DEVICE_H_
