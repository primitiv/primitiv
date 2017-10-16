from primitiv import trainers as T
from primitiv import Trainer

import unittest
import tempfile


class TrainerVirtualFuncTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sgd_virtual(self):
        t = T.SGD()
        self.assertEqual(t.name(), "SGD")
        uint_configs = {b'Trainer.epoch': 1}
        float_configs = {b'SGD.eta': 0.0,
                         b'Trainer.clip_threshold': 0.0,
                         b'Trainer.lr_scale': 1.0,
                         b'Trainer.l2_strength': 0.0,
        }
        t.set_configs(uint_configs, float_configs)
        uint_configs, float_configs = t.get_configs()
        self.assertEqual(uint_configs[b'Trainer.epoch'], 1)

    def test_momentum_sgd_virtual(self):
        t = T.MomentumSGD()
        self.assertEqual(t.name(), "MomentumSGD")
        uint_configs = {b'Trainer.epoch': 1}
        float_configs = {b'MomentumSGD.momentum': 1.0,
                         b'MomentumSGD.eta': 0.0,
                         b'Trainer.clip_threshold': 0.0,
                         b'Trainer.lr_scale': 1.0,
                         b'Trainer.l2_strength': 0.0,
        }
        t.set_configs(uint_configs, float_configs)
        uint_configs, float_configs = t.get_configs()
        self.assertEqual(uint_configs[b'Trainer.epoch'], 1)

    def test_adagrad_virtual(self):
        t = T.AdaGrad()
        self.assertEqual(t.name(), "AdaGrad")
        uint_configs = {b'Trainer.epoch': 1}
        float_configs = {b'AdaGrad.eps': 0.0,
                         b'AdaGrad.eta': 0.0,
                         b'Trainer.clip_threshold': 0.0,
                         b'Trainer.lr_scale': 1.0,
                         b'Trainer.l2_strength': 0.0,
        }
        t.set_configs(uint_configs, float_configs)
        uint_configs, float_configs = t.get_configs()
        self.assertEqual(uint_configs[b'Trainer.epoch'], 1)

    def test_adam_virtual(self):
        t = T.Adam()
        self.assertEqual(t.name(), "Adam")
        uint_configs = {b'Trainer.epoch': 1}
        float_configs = {b'Trainer.lr_scale': 1.0,
                         b'Adam.beta2': 1.0,
                         b'Adam.eps': 0.0,
                         b'Trainer.clip_threshold': 0.0,
                         b'Adam.alpha': 0.0,
                         b'Trainer.l2_strength': 0.0,
                         b'Adam.beta1': 1.0,
        }
        t.set_configs(uint_configs, float_configs)
        uint_configs, float_configs = t.get_configs()
        self.assertEqual(uint_configs[b'Trainer.epoch'], 1)
