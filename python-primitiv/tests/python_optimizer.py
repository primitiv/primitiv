from primitiv import optimizers as T
from primitiv import Optimizer, Parameter, Device, Graph, Shape
from primitiv import initializers as I
from primitiv import devices as D
from primitiv import operators as F
from primitiv import tensor_operators as tF

import unittest
import tempfile

import numpy as np


class TestAdam(Optimizer):
    def __init__(self, alpha, beta1, beta2, eps):
        super().__init__()
        self.alpha_ = np.float32(alpha)
        self.beta1_ = np.float32(beta1)
        self.beta2_ = np.float32(beta2)
        self.eps_ = np.float32(eps)

    def configure_parameter(self, param):
        for name in ("testadam-m1", "testadam-m2"):
            if name not in param.stats:
                param.add_stats(name, param.shape())
                param.stats[name].reset(0)

    def update_parameter(self, scale, param):
        epoch = self.get_epoch() + 1
        g = param.gradient
        param.stats["testadam-m1"] = self.beta1_ * param.stats["testadam-m1"] + (1 - self.beta1_) * g
        param.stats["testadam-m2"] = self.beta2_ * param.stats["testadam-m2"] + (1 - self.beta2_) * g * g
        mm1 = param.stats["testadam-m1"] / (1 - self.beta1_ ** epoch)
        mm2 = param.stats["testadam-m2"] / (1 - self.beta2_ ** epoch)
        param.value -= (scale * self.alpha_) * mm1 / (tF.sqrt(mm2) + self.eps_)

    def get_configs(self):
        uint_configs = {}
        float_configs = {
            "TestAdam.alpha": self.alpha_,
            "TestAdam.beta1": self.beta1_,
            "TestAdam.beta2": self.beta2_,
            "TestAdam.eps": self.eps_,
        }
        return uint_configs, float_configs

    def set_configs(self, uint_configs, float_configs):
        self.alpha_ = float_configs["TestAdam.alpha"]
        self.beta1_ = float_configs["TestAdam.beta1"]
        self.beta2_ = float_configs["TestAdam.beta2"]
        self.eps_ = float_configs["TestAdam.eps"]


class TestException(Exception):
    pass


class ExceptionOptimizer(Optimizer):

    def configure_parameter(self, param):
        raise TestException("configure_parameter")

    def update_parameter(self, scale, param):
        raise TestException("update_parameter")

    def get_configs(self):
        raise TestException("get_configs")

    def set_configs(self, uint_configs, float_configs):
        raise TestException("set_configs")


class IncompleteOptimizer(Optimizer):
    pass


def train_func(optimizer):
    dev = D.Naive(12345)
    Device.set_default(dev)
    g = Graph()
    Graph.set_default(g)

    pw1 = Parameter([8, 2], I.XavierUniform())
    pb1 = Parameter([8], I.Constant(0))
    pw2 = Parameter([1, 8], I.XavierUniform())
    pb2 = Parameter([1], I.Constant(0))

    optimizer.add_parameter(pw1)
    optimizer.add_parameter(pb1)
    optimizer.add_parameter(pw2)
    optimizer.add_parameter(pb2)

    input_data = [1, 1, 1, -1, -1, 1, -1, -1]
    output_data = [1, -1, -1, 1]

    for i in range(10):
        g.clear()
        x = F.raw_input(Shape([2], 4), input_data)
        w1 = F.parameter(pw1)
        b1 = F.parameter(pb1)
        w2 = F.parameter(pw2)
        b2 = F.parameter(pb2)
        h = F.tanh(w1 @ x + b1)
        y = w2 @ h + b2

        t = F.raw_input(Shape([], 4), output_data)
        diff = t - y
        loss = F.batch.mean(diff * diff)

        optimizer.reset_gradients()
        loss.backward()
        optimizer.update()

    return [pw1.value.to_list(),
            pb1.value.to_list(),
            pw2.value.to_list(),
            pb2.value.to_list()]


class PythonOptimizerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.t = TestAdam(alpha = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8)

    def tearDown(self):
        pass

    def test_pyoptimizer_get_set_config(self):
        uint_configs, float_configs = Optimizer.get_configs(self.t)
        self.assertAlmostEqual(uint_configs['Optimizer.epoch'], 0)
        self.assertAlmostEqual(float_configs['TestAdam.alpha'], 0.001)
        self.assertAlmostEqual(float_configs['TestAdam.beta1'], 0.9)
        self.assertAlmostEqual(float_configs['TestAdam.beta2'], 0.999)
        self.assertAlmostEqual(float_configs['TestAdam.eps'], 1e-8, places=10)
        float_configs['TestAdam.beta1'] = 200
        Optimizer.set_configs(self.t, uint_configs, float_configs)
        self.assertEqual(self.t.beta1_, 200)

    def test_pyoptimizer_parameter(self):
        dev = D.Naive()
        Device.set_default(dev)
        pw1 = Parameter([8, 2], I.XavierUniform())
        self.t.add_parameter(pw1)
        self.assertIn("testadam-m1", pw1.stats)
        self.assertIn("testadam-m2", pw1.stats)

    def test_pyoptimizer_compare_with_cpp(self):
        c_optimizer = T.Adam(alpha = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8)
        py_params = train_func(self.t)
        c_params = train_func(c_optimizer)
        py_uint_configs, py_float_configs = Optimizer.get_configs(self.t)
        c_uint_configs, c_float_configs = c_optimizer.get_configs()
        self.assertEqual(py_uint_configs["Optimizer.epoch"], c_uint_configs["Optimizer.epoch"])
        self.assertEqual(py_float_configs["TestAdam.alpha"], c_float_configs["Adam.alpha"])
        self.assertEqual(py_float_configs["TestAdam.beta1"], c_float_configs["Adam.beta1"])
        self.assertEqual(py_float_configs["TestAdam.beta2"], c_float_configs["Adam.beta2"])
        self.assertEqual(py_float_configs["TestAdam.eps"], c_float_configs["Adam.eps"])
        self.assertEqual(py_float_configs["Optimizer.clip_threshold"], c_float_configs["Optimizer.clip_threshold"])
        self.assertEqual(py_float_configs["Optimizer.l2_strength"], c_float_configs["Optimizer.l2_strength"])
        self.assertEqual(py_float_configs["Optimizer.lr_scale"], c_float_configs["Optimizer.lr_scale"])
        self.assertTrue(np.isclose(py_params[0], c_params[0]).all())
        self.assertTrue(np.isclose(py_params[1], c_params[1]).all())
        self.assertTrue(np.isclose(py_params[2], c_params[2]).all())
        self.assertTrue(np.isclose(py_params[3], c_params[3]).all())

    def test_pyoptimizer_loadsave(self):
        t_loaded = TestAdam(alpha = 0, beta1 = 0, beta2 = 0, eps = 0)
        self.assertEqual(t_loaded.alpha_, 0)
        self.assertEqual(t_loaded.beta1_, 0)
        self.assertEqual(t_loaded.beta2_, 0)
        self.assertEqual(t_loaded.eps_, 0)
        with tempfile.NamedTemporaryFile() as fp:
            self.t.save(fp.name)
            t_loaded.load(fp.name)
        self.assertAlmostEqual(t_loaded.alpha_, 0.001)
        self.assertAlmostEqual(t_loaded.beta1_, 0.9)
        self.assertAlmostEqual(t_loaded.beta2_, 0.999)
        self.assertAlmostEqual(t_loaded.eps_, 1e-8, places=10)

    def test_pyoptimizer_propagate_exception(self):
        dev = D.Naive()
        Device.set_default(dev)
        optimizer = ExceptionOptimizer()
        p = Parameter()
        with self.assertRaises(TestException) as ctx:
            optimizer.add_parameter(p)
        self.assertEqual(str(ctx.exception), "configure_parameter")
        with self.assertRaises(TestException) as ctx:
            optimizer.update()
        self.assertEqual(str(ctx.exception), "update_parameter")
        with self.assertRaises(TestException) as ctx:
            Optimizer.get_configs(optimizer)
        self.assertEqual(str(ctx.exception), "get_configs")
        with self.assertRaises(TestException) as ctx:
            Optimizer.set_configs(optimizer, {'Optimizer.epoch': 1},
                                         {'Optimizer.clip_threshold': 0.0,
                                          'Optimizer.lr_scale': 1.0,
                                          'Optimizer.l2_strength': 0.0})
        self.assertEqual(str(ctx.exception), "set_configs")


    def test_pyoptimizer_not_implemented(self):
        dev = D.Naive()
        Device.set_default(dev)
        optimizer = IncompleteOptimizer()
        p = Parameter()
        with self.assertRaises(NotImplementedError):
            optimizer.add_parameter(p)
        with self.assertRaises(NotImplementedError):
            optimizer.update()
        with self.assertRaises(NotImplementedError):
            Optimizer.get_configs(optimizer)
        with self.assertRaises(NotImplementedError):
            Optimizer.set_configs(optimizer, {'Optimizer.epoch': 1},
                                         {'Optimizer.clip_threshold': 0.0,
                                          'Optimizer.lr_scale': 1.0,
                                          'Optimizer.l2_strength': 0.0})
