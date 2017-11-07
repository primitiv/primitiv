from primitiv import Device, Graph, Shape, Parameter
from primitiv.devices import Naive
from primitiv import tensor_operators as tF
from primitiv import operators as F

try:
    from primitiv.devices import CUDA
except ImportError:
    CUDA = None

import unittest
from test.support import captured_stderr


class DeviceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.naive_dev = Naive()
        self.cuda_dev = CUDA(0) if CUDA else None
        self.graph = Graph()

    def tearDown(self):
        pass

    def test_default_device(self):
        Device.set_default(self.naive_dev)
        dev = Device.get_default()
        self.assertIs(self.naive_dev, dev)
        if self.cuda_dev:
            Device.set_default(self.cuda_dev)
            dev = Device.get_default()
            self.assertIs(self.cuda_dev, dev)

    def test_num_devices(self):
        if CUDA:
            self.assertIsInstance(CUDA.num_devices(), int)

    def test_device_instance(self):
        Device.set_default(self.naive_dev)
        Graph.set_default(self.graph)

        dev = Device.get_default()
        self.assertIs(dev, self.naive_dev)

        tensor = tF.input([0], Shape([]))
        dev = tensor.device()
        self.assertIs(dev, self.naive_dev)

        node = F.input([0], Shape([]))
        dev = node.device()
        self.assertIs(dev, self.naive_dev)

        my_device = Naive()
        self.assertIsNot(my_device, self.naive_dev)

        node = F.input([0], Shape([]), device=my_device)
        dev = node.device()
        self.assertIs(dev, my_device)

        dev = self.graph.get_device(node)
        self.assertIs(dev, my_device)

        param = Parameter([], [1])
        dev = param.device()
        self.assertIs(dev, self.naive_dev)
