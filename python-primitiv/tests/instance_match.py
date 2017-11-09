from primitiv import Device
from primitiv import Graph
from primitiv import Parameter
from primitiv import Shape
from primitiv import Tensor
from primitiv import initializers as I
from primitiv import operators as F
from primitiv import tensor_operators as tF
from primitiv.devices import Naive

import numpy as np
import unittest


class ArgumentTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.device = Naive()
        Device.set_default(self.device)
        self.graph = Graph()
        Graph.set_default(self.graph)

    def tearDown(self):
        pass

    def test_device_instance(self):
        dev = Device.get_default()
        self.assertIs(dev, self.device)

        tensor = tF.raw_input([], [0])
        #dev = tensor.device()
        #self.assertIs(dev, self.device)

        node = F.raw_input([], [0])
        dev = node.device()
        self.assertIs(dev, self.device)

        my_device = Naive()
        self.assertIsNot(my_device, self.device)

        node = F.raw_input([], [0], device=my_device)
        dev = node.device()
        self.assertIs(dev, my_device)

        dev = self.graph.get_device(node)
        self.assertIs(dev, my_device)

        param = Parameter([], I.Constant(1))
        dev = param.device()
        self.assertIs(dev, self.device)

    def test_graph_instance(self):
        g = Graph.get_default()
        self.assertIs(g, self.graph)

        node = F.raw_input([], [0])
        g = node.graph()
        self.assertIs(g, self.graph)

    def test_tensor_instance(self):
        param = Parameter([], I.Constant(1))
        t_origin = param.gradient
        t = param.gradient
        self.assertIs(t, t_origin)

        t = Tensor(t_origin)
        self.assertEqual(t.to_list(), t.to_list())
        self.assertIsNot(t, t_origin)

        t = t_origin
        t *= 2
        self.assertIs(t, t_origin)

        t = t * 2
        self.assertIsNot(t, t_origin)
