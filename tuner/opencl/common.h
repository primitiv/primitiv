#ifndef PRIMITIV_TUNER_OPENCL_COMMON_H_
#define PRIMITIV_TUNER_OPENCL_COMMON_H_

#define PRIMITIV_TUNER_OPENCL_MAIN(tuner_class) \
  int main(const int argc, const char *argv[]) { \
    int platform_id; \
    int device_id; \
    std::string parameter_file; \
    try { \
      if (argc != 4) { \
        PRIMITIV_THROW_ERROR("expected 3 arguments") \
      } \
      parameter_file = argv[3]; \
      try { \
        platform_id = std::stoi(argv[1]); \
        device_id = std::stoi(argv[2]); \
      } catch (...) { \
        PRIMITIV_THROW_ERROR("platform-id and device-id must be integers") \
      } \
      if (platform_id < 0 || device_id < 0) { \
        PRIMITIV_THROW_ERROR( \
          "platform-id and device-id must be greater than or equal to 0") \
      } \
    } catch (const primitiv::Error &e) { \
      std::cerr << "invalid arguments: " \
        << e.what() << std::endl \
        << "usage: " << argv[0] \
        << " <platform-id> <device-id> <parameter-file>" << std::endl; \
      return 1; \
    } \
    primitiv::devices::OpenCLKernelParameters params; \
    try { \
      params.load(parameter_file); \
    } catch (...) { \
      std::cerr << "Could not open `" << parameter_file \
        << "`. Creating a new file." << std::endl; \
      params.save(parameter_file); \
    } \
    tuner_class tuner(platform_id, device_id); \
    try { \
      tuner.tune(params); \
    } catch (const primitiv::Error &err) { \
      std::cerr << err.what() << std::endl; \
      return 1; \
    } \
    params.save(parameter_file); \
    return 0; \
  }

#endif  // PRIMITIV_TUNER_OPENCL_COMMON_H_
