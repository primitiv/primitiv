from primitiv import Optimizer, Parameter, Device, Graph, Shape
from primitiv import initializers as I
from primitiv import devices as D
from primitiv import operators as F
from primitiv import tensor_operators as tF
from primitiv._parameter import ParameterStatistics

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
        Device.set_default(self.dev)
        self.p = Parameter([8], I.Constant(0))
        self.p.value.reset_by_vector([1, 2, 3, 4, 5, 6, 7, 8])

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
            self.p.stats = ParameterStatistics(self.p)

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
