from primitiv import Device
from primitiv import tensor_operators as tF
from primitiv.devices import Naive

import numpy as np
import unittest


class TensorOperatorsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.device = Naive()
        Device.set_default(self.device)
        self.a = np.array([[1, 2], [3, 4]], np.float32)
        self.b = np.array([[1, 1], [4, 8]], np.float32)

    def tearDown(self):
        pass

    def test_tensor_pos(self):
        x = tF.input(self.a)
        y = tF.input(self.b)
        self.assertTrue(((+x).to_ndarrays()[0] == self.a).all())

    def test_tensor_neg(self):
        x = tF.input(self.a)
        y = tF.input(self.b)
        self.assertTrue(((-x).to_ndarrays()[0] == -self.a).all())

    def test_tensor_add(self):
        x = tF.input(self.a)
        y = tF.input(self.b)
        self.assertTrue(((x + y).to_ndarrays()[0] == np.array([[2, 3], [7, 12]])).all())
        self.assertTrue(((x + 2).to_ndarrays()[0] == np.array([[3, 4], [5, 6]])).all())
        self.assertTrue(((2 + x).to_ndarrays()[0] == np.array([[3, 4], [5, 6]])).all())

    def test_tensor_sub(self):
        x = tF.input(self.a)
        y = tF.input(self.b)
        self.assertTrue(((x - y).to_ndarrays()[0] == np.array([[0, 1], [-1, -4]])).all())
        self.assertTrue(((x - 2).to_ndarrays()[0] == np.array([[-1, 0], [1, 2]])).all())
        self.assertTrue(((2 - x).to_ndarrays()[0] == np.array([[1, 0], [-1, -2]])).all())

    def test_tensor_mul(self):
        x = tF.input(self.a)
        y = tF.input(self.b)
        self.assertTrue(((x * y).to_ndarrays()[0] == np.array([[1, 2], [12, 32]])).all())
        self.assertTrue(((x * 2).to_ndarrays()[0] == np.array([[2, 4], [6, 8]])).all())
        self.assertTrue(((2 * x).to_ndarrays()[0] == np.array([[2, 4], [6, 8]])).all())

    def test_tensor_matmul(self):
        x = tF.input(self.a)
        y = tF.input(self.b)
        self.assertTrue(((x @ y).to_ndarrays()[0] == np.array([[9, 17], [19, 35]])).all())
        self.assertRaises(TypeError, lambda: x @ 2)
        self.assertRaises(TypeError, lambda: 2 @ x)

    def test_tensor_truediv(self):
        x = tF.input(self.a)
        y = tF.input(self.b)
        self.assertTrue(((x / y).to_ndarrays()[0] == np.array([[1, 2], [0.75, 0.5]])).all())
        self.assertTrue(((x / 2).to_ndarrays()[0] == np.array([[0.5, 1], [1.5, 2]])).all())
        self.assertTrue(((2 / y).to_ndarrays()[0] == np.array([[2, 2], [0.5, 0.25]])).all())

    def test_tensor_pow(self):
        x = tF.input(self.a)
        y = tF.input(self.b)
        self.assertTrue(np.isclose((x ** y).to_ndarrays()[0], np.array([[1, 2], [81, 65536]])).all())
        self.assertTrue(np.isclose((x ** 2).to_ndarrays()[0], np.array([[1, 4], [9, 16]])).all())
        self.assertTrue(np.isclose((2 ** x).to_ndarrays()[0], np.array([[2, 4], [8, 16]])).all())
        self.assertRaises(TypeError, lambda: pow(x, y, 2))

    def test_tensor_iadd(self):
        x = tF.input(self.a)
        y = tF.input(self.b)
        x_tmp = x
        x += y
        self.assertIs(x, x_tmp)
        self.assertTrue((x.to_ndarrays()[0] == np.array([[2, 3], [7, 12]])).all())

    def test_tensor_isub(self):
        x = tF.input(self.a)
        y = tF.input(self.b)
        x_tmp = x
        x -= y
        self.assertIs(x, x_tmp)
        self.assertTrue((x.to_ndarrays()[0] == np.array([[0, 1], [-1, -4]])).all())

    def test_tensor_imul(self):
        x = tF.input(self.a)
        x_tmp = x
        x *= 2
        self.assertIs(x, x_tmp)
        self.assertTrue((x.to_ndarrays()[0] == np.array([[2, 4], [6, 8]])).all())
