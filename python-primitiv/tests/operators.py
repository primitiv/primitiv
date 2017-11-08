from primitiv import Device
from primitiv import Graph
from primitiv import Shape
from primitiv import operators as F
from primitiv.devices import Naive

import numpy as np
import unittest


class OperatorTest(unittest.TestCase):

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
        self.ndarray_data = [
            np.array([
                [ 1, 2, 3],
                [ 4, 5, 6],
                [ 7, 8, 9],
                [10,11,12],
            ], np.float32),
            np.array([
                [13,14,15],
                [16,17,18],
                [19,20,21],
                [22,23,24],
            ], np.float32),
        ]
        self.list_data = [
             1.0,  4.0,  7.0, 10.0,  2.0,  5.0,  8.0, 11.0,  3.0,  6.0,  9.0, 12.0,
            13.0, 16.0, 19.0, 22.0, 14.0, 17.0, 20.0, 23.0, 15.0, 18.0, 21.0, 24.0,
        ]

    def tearDown(self):
        pass

    def test_operators_input_argument(self):
        # list[ndarray] w/o shape
        x = F.input(self.ndarray_data)
        self.assertEqual(x.to_list(), self.list_data)
        self.assertEqual(x.shape(), Shape([4, 3], 2))

        # list[ndarray] w/ shape
        x = F.input(self.ndarray_data, Shape([2, 3], 4))
        self.assertEqual(x.to_list(), self.list_data)
        self.assertEqual(x.shape(), Shape([2, 3], 4))

        # ndarray w/o shape
        x = F.input(self.ndarray_data[0])
        self.assertEqual(x.to_list(), self.list_data[:12])
        self.assertEqual(x.shape(), Shape([4, 3], 1))

        # ndarray w/ shape
        x = F.input(self.ndarray_data[0], Shape([2, 3], 2))
        self.assertEqual(x.to_list(), self.list_data[:12])
        self.assertEqual(x.shape(), Shape([2, 3], 2))

        # list[float] w/o shape
        self.assertRaises(TypeError, lambda: F.input(self.list_data))

        # list[float] w/ shape
        x = F.input(self.list_data, shape=Shape([4, 3], 2))
        self.assertEqual(x.to_list(), self.list_data)
        self.assertEqual(x.shape(), Shape([4, 3], 2))

    def test_input_ndarrays(self):
        x = F.input(self.ndarray_data)
        self.assertEqual(x.to_list(), self.list_data)
        self.assertTrue((x.to_ndarrays()[0] == self.ndarray_data[0]).all())
        self.assertTrue((x.to_ndarrays()[1] == self.ndarray_data[1]).all())
