from primitiv._device cimport Device


cdef class OpenCL(Device):

    def __init__(self, unsigned platform_id, unsigned device_id, rng_seed=None):
        """Creates a new OpenCL device.

        :param platform_id: Platform ID.
        :type rng_seed: int
        :param device_id: Device ID on the selected platform.
        :type rng_seed: int
        :param rng_seed: Seed value of the random number generator.
        :type rng_seed: int or None

        """
        if self.wrapped is not NULL:
            raise TypeError("__init__() has already been called.")
        if rng_seed is None:
            self.wrapped = new CppOpenCL(platform_id, device_id)
        else:
            self.wrapped = new CppOpenCL(platform_id, device_id, <unsigned> rng_seed)

        Device.register_wrapper(self.wrapped, self)

    def __dealloc__(self):
        if self.wrapped is not NULL:
            del self.wrapped
            self.wrapped = NULL

    @staticmethod
    def num_platforms():
        """Retrieves the number of active platforms.

        :return: Number of active platforms.
        :rtype: int

        """
        return CppOpenCL.num_platforms()

    @staticmethod
    def num_devices(unsigned platform_id):
        """Retrieves the number of active hardwares.

        :param platform_id: Platform ID.
        :type platform_id: int
        :return: Number of active hardwares.
        :rtype: int

        """
        return CppOpenCL.num_devices(platform_id)
