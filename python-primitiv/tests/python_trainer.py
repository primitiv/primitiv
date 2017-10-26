from primitiv import trainers as T
from primitiv import Trainer, Parameter, Device, Graph, Shape
from primitiv import initializers as I
from primitiv import devices as D
from primitiv import operators as F
from primitiv import tensor_operators as tF

import unittest
import tempfile

import numpy as np


class TestAdam(Trainer):
    def __init__(self, alpha, beta1, beta2, eps):
        super().__init__()
        self.alpha_ = alpha
        self.beta1_ = beta1
        self.beta2_ = beta2
        self.eps_ = eps

    def name(self):
        return "MyAdam"

    def configure_parameter(self, param):
        for name in ("adam-m1", "adam-m2"):
            if name not in param.stats:
                param.add_stats(name, param.shape())
                param.stats[name].reset(0)

    def update_parameter(self, scale, param):
        epoch = self.get_epoch() + 1
        g = param.gradient()
        param.stats["adam-m1"] = self.beta1_ * param.stats["adam-m1"] + (1 - self.beta1_) * g
        param.stats["adam-m2"] = self.beta2_ * param.stats["adam-m2"] + (1 - self.beta2_) * g * g
        mm1 = param.stats["adam-m1"] / (1 - self.beta1_ ** epoch)
        mm2 = param.stats["adam-m2"] / (1 - self.beta2_ ** epoch)
        param.value -= (scale * self.alpha_) * mm1 / (tF.sqrt(mm2) + self.eps_)

    def get_configs(self):
        uint_configs = {}
        float_configs = {
            "Adam.alpha": self.alpha_,
            "Adam.beta1": self.beta1_,
            "Adam.beta2": self.beta2_,
            "Adam.eps": self.eps_,
        }
        return uint_configs, float_configs

    def set_configs(self, uint_configs, float_configs):
        self.alpha_ = float_configs["Adam.alpha"]
        self.beta1_ = float_configs["Adam.beta1"]
        self.beta2_ = float_configs["Adam.beta2"]
        self.eps_ = float_configs["Adam.eps"]


class TestException(Exception):
    pass


class ExceptionTrainer(Trainer):

    def name(self):
        raise TestException("name")

    def configure_parameter(self, param):
        raise TestException("configure_parameter")

    def update_parameter(self, scale, param):
        raise TestException("update_parameter")

    def get_configs(self):
        raise TestException("get_configs")

    def set_configs(self, uint_configs, float_configs):
        raise TestException("set_configs")


class IncompleteTrainer(Trainer):
    pass


def train_func(trainer):
    dev = D.Naive(12345)
    Device.set_default(dev)
    g = Graph()
    Graph.set_default(g)

    pw1 = Parameter("w1", [8, 2], I.XavierUniform())
    pb1 = Parameter("b1", [8], I.Constant(0))
    pw2 = Parameter("w2", [1, 8], I.XavierUniform())
    pb2 = Parameter("b2", [1], I.Constant(0))

    trainer.add_parameter(pw1)
    trainer.add_parameter(pb1)
    trainer.add_parameter(pw2)
    trainer.add_parameter(pb2)

    input_data = [1, 1, 1, -1, -1, 1, -1, -1]
    output_data = [1, -1, -1, 1]

    for i in range(10):
        g.clear()
        x = F.input(input_data, Shape([2], 4))
        w1 = F.parameter(pw1)
        b1 = F.parameter(pb1)
        w2 = F.parameter(pw2)
        b2 = F.parameter(pb2)
        h = F.tanh(w1 @ x + b1)
        y = w2 @ h + b2

        t = F.input(output_data, Shape([], 4))
        diff = t - y
        loss = F.batch.mean(diff * diff)

        trainer.reset_gradients()
        loss.backward()
        trainer.update()

    return [pw1.value.to_list(),
            pb1.value.to_list(),
            pw2.value.to_list(),
            pb2.value.to_list()]


class PythonTrainerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.t = TestAdam(alpha = 0.001,  beta1 = 0.9, beta2 = 0.999, eps = 1e-8)

    def tearDown(self):
        pass

    def test_pytrainer_get_set_config(self):
        uint_configs, float_configs = Trainer.get_configs(self.t)
        self.assertAlmostEqual(uint_configs['Trainer.epoch'], 0)
        self.assertAlmostEqual(float_configs['Adam.alpha'], 0.001)
        self.assertAlmostEqual(float_configs['Adam.beta1'], 0.9)
        self.assertAlmostEqual(float_configs['Adam.beta2'], 0.999)
        self.assertAlmostEqual(float_configs['Adam.eps'], 1e-8, places=10)
        float_configs['Adam.beta1'] = 200
        Trainer.set_configs(self.t, uint_configs, float_configs)
        self.assertEqual(self.t.beta1_, 200)

    def test_pytrainer_parameter(self):
        dev = D.Naive()
        Device.set_default(dev)
        pw1 = Parameter("w1", [8, 2], I.XavierUniform())
        self.t.add_parameter(pw1)
        self.assertIn("adam-m1", pw1.stats)
        self.assertIn("adam-m2", pw1.stats)

    def test_pytrainer_compare_with_cpp(self):
        c_trainer = T.Adam(alpha = 0.001,  beta1 = 0.9, beta2 = 0.999, eps = 1e-8)
        py_params = train_func(self.t)
        c_params = train_func(c_trainer)
        self.assertEqual(Trainer.get_configs(self.t), c_trainer.get_configs())
        self.assertTrue(np.isclose(py_params[0], c_params[0]).all())
        self.assertTrue(np.isclose(py_params[1], c_params[1]).all())
        self.assertTrue(np.isclose(py_params[2], c_params[2]).all())
        self.assertTrue(np.isclose(py_params[3], c_params[3]).all())

    def test_pytrainer_name(self):
        self.assertEqual(Trainer.name(self.t), "MyAdam")

    def test_pytrainer_loadsave(self):
        t_loaded = TestAdam(alpha = 0,  beta1 = 0, beta2 = 0, eps = 0)
        self.assertEqual(t_loaded.alpha_, 0)
        self.assertEqual(t_loaded.beta1_, 0)
        self.assertEqual(t_loaded.beta2_, 0)
        self.assertEqual(t_loaded.eps_, 0)
        with tempfile.NamedTemporaryFile() as fp:
            self.t.save(fp.name)
            t_loaded.set_configs_by_file(fp.name)
        self.assertAlmostEqual(t_loaded.alpha_, 0.001)
        self.assertAlmostEqual(t_loaded.beta1_, 0.9)
        self.assertAlmostEqual(t_loaded.beta2_, 0.999)
        self.assertAlmostEqual(t_loaded.eps_, 1e-8, places=10)

    def test_pytrainer_propagate_exception(self):
        dev = D.Naive()
        Device.set_default(dev)
        trainer = ExceptionTrainer()
        with self.assertRaises(TestException) as ctx:
            Trainer.name(trainer)
        self.assertEqual(str(ctx.exception), "name")
        p = Parameter("p", Shape([]))
        with self.assertRaises(TestException) as ctx:
            trainer.add_parameter(p)
        self.assertEqual(str(ctx.exception), "configure_parameter")
        with self.assertRaises(TestException) as ctx:
            trainer.update()
        self.assertEqual(str(ctx.exception), "update_parameter")
        with self.assertRaises(TestException) as ctx:
            Trainer.get_configs(trainer)
        self.assertEqual(str(ctx.exception), "get_configs")
        with self.assertRaises(TestException) as ctx:
            Trainer.set_configs(trainer, {'Trainer.epoch': 1},
                                         {'Trainer.clip_threshold': 0.0,
                                          'Trainer.lr_scale': 1.0,
                                          'Trainer.l2_strength': 0.0})
        self.assertEqual(str(ctx.exception), "set_configs")


    def test_pytrainer_not_implemented(self):
        dev = D.Naive()
        Device.set_default(dev)
        trainer = IncompleteTrainer()
        with self.assertRaises(NotImplementedError):
            Trainer.name(trainer)
        p = Parameter("p", Shape([]))
        with self.assertRaises(NotImplementedError):
            trainer.add_parameter(p)
        with self.assertRaises(NotImplementedError):
            trainer.update()
        with self.assertRaises(NotImplementedError):
            Trainer.get_configs(trainer)
        with self.assertRaises(NotImplementedError):
            Trainer.set_configs(trainer, {'Trainer.epoch': 1},
                                         {'Trainer.clip_threshold': 0.0,
                                          'Trainer.lr_scale': 1.0,
                                          'Trainer.l2_strength': 0.0})
