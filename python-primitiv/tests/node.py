from primitiv import Device
from primitiv import Graph
from primitiv import operators as F
from primitiv.devices import Naive

import numpy as np
import unittest


class NodeOperatorsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.device = Naive()
        self.graph = Graph()
        Device.set_default(self.device)
        Graph.set_default(self.graph)
        self.a = np.array([[1, 2], [3, 4]], np.float32)
        self.b = np.array([[1, 1], [4, 8]], np.float32)

    def tearDown(self):
        pass

    def test_node_pos(self):
        x = F.input(self.a)
        y = F.input(self.b)
        self.assertTrue(((+x).to_ndarrays()[0] == self.a).all())

    def test_node_neg(self):
        x = F.input(self.a)
        y = F.input(self.b)
        self.assertTrue(((-x).to_ndarrays()[0] == -self.a).all())

    def test_node_add(self):
        x = F.input(self.a)
        y = F.input(self.b)
        self.assertTrue(((x + y).to_ndarrays()[0] == np.array([[2, 3], [7, 12]])).all())
        self.assertTrue(((x + 2).to_ndarrays()[0] == np.array([[3, 4], [5, 6]])).all())
        self.assertTrue(((2 + x).to_ndarrays()[0] == np.array([[3, 4], [5, 6]])).all())

    def test_node_sub(self):
        x = F.input(self.a)
        y = F.input(self.b)
        self.assertTrue(((x - y).to_ndarrays()[0] == np.array([[0, 1], [-1, -4]])).all())
        self.assertTrue(((x - 2).to_ndarrays()[0] == np.array([[-1, 0], [1, 2]])).all())
        self.assertTrue(((2 - x).to_ndarrays()[0] == np.array([[1, 0], [-1, -2]])).all())

    def test_node_mul(self):
        x = F.input(self.a)
        y = F.input(self.b)
        self.assertTrue(((x * y).to_ndarrays()[0] == np.array([[1, 2], [12, 32]])).all())
        self.assertTrue(((x * 2).to_ndarrays()[0] == np.array([[2, 4], [6, 8]])).all())
        self.assertTrue(((2 * x).to_ndarrays()[0] == np.array([[2, 4], [6, 8]])).all())

    def test_node_matmul(self):
        x = F.input(self.a)
        y = F.input(self.b)
        self.assertTrue(((x @ y).to_ndarrays()[0] == np.array([[9, 17], [19, 35]])).all())
        self.assertRaises(TypeError, lambda: x @ 2)
        self.assertRaises(TypeError, lambda: 2 @ x)

    def test_node_truediv(self):
        x = F.input(self.a)
        y = F.input(self.b)
        self.assertTrue(((x / y).to_ndarrays()[0] == np.array([[1, 2], [0.75, 0.5]])).all())
        self.assertTrue(((x / 2).to_ndarrays()[0] == np.array([[0.5, 1], [1.5, 2]])).all())
        self.assertTrue(((2 / y).to_ndarrays()[0] == np.array([[2, 2], [0.5, 0.25]])).all())

    def test_node_pow(self):
        x = F.input(self.a)
        y = F.input(self.b)
        self.assertTrue(np.isclose((x ** y).to_ndarrays()[0], np.array([[1, 2], [81, 65536]])).all())
        self.assertTrue(np.isclose((x ** 2).to_ndarrays()[0], np.array([[1, 4], [9, 16]])).all())
        self.assertTrue(np.isclose((2 ** x).to_ndarrays()[0], np.array([[2, 4], [8, 16]])).all())
        self.assertTrue(np.isclose((x ** -2).to_ndarrays()[0], np.array([[1, 1/4], [1/9, 1/16]])).all())
        input_arr = np.array([1, -1, 3, -3, 5, -5])
        x = F.input(input_arr)
        self.assertTrue(((x ** 6).to_ndarrays()[0] == np.array([1, 1, 729, 729, 15625, 15625])).all())
        self.assertTrue(((x ** 9).to_ndarrays()[0] == np.array([1, -1, 19683, -19683, 1953125, -1953125])).all())
        input_arr = np.array([1, -1])
        x = F.input(input_arr)
        self.assertTrue(((x ** 0x7fffffff).to_ndarrays()[0] == np.array([1, -1])).all())
        self.assertTrue(((x ** -0x80000000).to_ndarrays()[0] == np.array([1, 1])).all())
        self.assertTrue(np.isnan((x ** 0x80000000).to_ndarrays()[0]).any())
        self.assertTrue(np.isnan((x ** -0x80000001).to_ndarrays()[0]).any())
        self.assertRaises(TypeError, lambda: pow(x, y, 2))
