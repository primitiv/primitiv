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

private:
  std::map<void *, unsigned> blocks_;
};

}  // namespace primitiv

#endif  // PRIMITIV_CPU_DEVICE_H_
