from primitiv import DefaultScope

from primitiv import CPUDevice
from primitiv import Graph
from primitiv import operators as F

import numpy as np

import unittest


class ArrayOrderingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.device = CPUDevice()
        self.graph = Graph()
        self.input_data = [
            np.array([
                [ 1, 2, 3],
                [ 4, 5, 6],
                [ 7, 8, 9],
            ], np.float32),
            np.array([
                [11,12,13],
                [14,15,16],
                [17,18,19],
            ], np.float32),
        ]
        self.list_expected = [
             1.0,  4.0,  7.0,  2.0,  5.0,  8.0,  3.0,  6.0,  9.0,
            11.0, 14.0, 17.0, 12.0, 15.0, 18.0, 13.0, 16.0, 19.0,
        ]

    def tearDown(self):
        pass

    def test_input_ndarrays(self):
        with DefaultScope(self.device):
            with DefaultScope(self.graph):
                x = F.input(self.input_data)
                self.assertEqual(x.to_list(), self.list_expected)
                self.assertTrue((x.to_ndarrays()[0] == self.input_data[0]).all())
                self.assertTrue((x.to_ndarrays()[1] == self.input_data[1]).all())
