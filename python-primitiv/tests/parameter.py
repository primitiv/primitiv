from primitiv import trainers as T
from primitiv import Trainer, Parameter, Device, Graph, Shape
from primitiv import initializers as I
from primitiv import devices as D
from primitiv import operators as F
from primitiv import tensor_operators as tF
from primitiv._parameter import _ParameterStatistics

import unittest

import numpy as np


class ParameterTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.dev = D.Naive()
        self.graph = Graph()
        Device.set_default(self.dev)
        Graph.set_default(self.graph)
        self.p = Parameter(initializer=np.array([1, 2, 3, 4, 5, 6, 7, 8]))
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

    def test_parameter_stats(self):
        self.p.add_stats("stat1", Shape([2, 3]))
        self.p.add_stats("stat2", Shape([2, 4]))
        st1 = self.p.stats["stat1"]
        st1.reset(0)
        self.assertTrue((st1.to_ndarrays()[0] == np.zeros([2, 3])).all())
        self.p.stats["stat1"] = tF.input(np.ones([2, 3]))
        self.assertTrue((st1.to_ndarrays()[0] == np.ones([2, 3])).all())
        self.assertIn("stat1", self.p.stats)
        self.assertIn("stat2", self.p.stats)
        self.assertNotIn("stat3", self.p.stats)
        with self.assertRaises(NotImplementedError):
            del self.p.stats["stat1"]
        with self.assertRaises(AttributeError):
            self.p.stats = _ParameterStatistics(self.p)

    def test_parameter_value(self):
        self.assertTrue((self.p.value.to_ndarrays() == np.array([1, 2, 3, 4, 5, 6, 7, 8])).all())
        val = self.p.value
        self.p.value += tF.input(np.ones([8]))
        self.assertTrue((val.to_ndarrays()[0] == np.array([2, 3, 4, 5, 6, 7, 8, 9])).all())
        with self.assertRaises(NotImplementedError):
            del self.p.value

    def test_parameter_gradient(self):
        self.p.reset_gradient()
        self.assertTrue((self.p.gradient.to_ndarrays() == np.zeros([8])).all())
        grad = self.p.gradient
        self.p.gradient += tF.input(np.ones([8]))
        self.assertTrue((grad.to_ndarrays()[0] == np.ones([8])).all())
        with self.assertRaises(NotImplementedError):
            del self.p.gradient

    def test_parameter_argument(self):
        # w/o arguments
        p = Parameter()
        self.assertFalse(p.valid())

        # shape w/ Initializer
        p = Parameter(Shape([4, 3]), I.Constant(1))
        self.assertEqual(p.shape(), Shape([4, 3]))
        self.assertEqual(p.value.to_list(), [1] * 12)

        # shape w/ list[float]
        p = Parameter(Shape([4, 3]), self.list_data[:12])
        self.assertEqual(p.shape(), Shape([4, 3]))
        self.assertEqual(p.value.to_list(), self.list_data[:12])

        # ndarray w/o shape
        p = Parameter(initializer=self.ndarray_data[0])
        self.assertEqual(p.shape(), Shape([4, 3]))
        self.assertEqual(p.value.to_list(), self.list_data[:12])

        # ndarray w/ shape
        p = Parameter(Shape([2, 6]), self.ndarray_data[0])
        self.assertEqual(p.shape(), Shape([2, 6]))
        self.assertEqual(p.value.to_list(), self.list_data[:12])

        # list[float] w/o shape
        self.assertRaises(TypeError, lambda: Parameter(initializer=self.list_data[:12]))
