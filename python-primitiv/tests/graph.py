from primitiv import Device
from primitiv import Graph
from primitiv import Shape
from primitiv import operators as F
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

    def test_graph_instance(self):
        g = Graph.get_default()
        self.assertIs(g, self.graph)

        node = F.input([0], Shape([]))
        g = node.graph()
        self.assertIs(g, self.graph)
