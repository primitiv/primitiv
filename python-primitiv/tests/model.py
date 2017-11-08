from primitiv import Trainer, Parameter, Device, Model
from primitiv import devices as D
from primitiv._model import _ModelParameter
from primitiv._model import _ModelSubModel

import unittest


class ModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.dev = D.Naive()
        Device.set_default(self.dev)

    def tearDown(self):
         pass

    def test_model_parameter(self):
        model = Model()
        param = Parameter()
        model.add_parameter("p", param)
        self.assertIs(model.params["p"], param)

    def test_model_parameter_deep(self):
        model1 = Model()
        model2 = Model()
        model1.add_submodel("m2", model2)
        model3 = Model()
        model2.add_submodel("m3", model3)
        param = Parameter()
        model3.add_parameter("p", param)
        self.assertIs(model1.params["m2", "m3", "p"], param)

    def test_model_submodel(self):
        model1 = Model()
        model2 = Model()
        model1.add_submodel("m", model2)
        self.assertIs(model1.submodels["m"], model2)

    def test_model_submodel_deep(self):
        model1 = Model()
        model2 = Model()
        model1.add_submodel("m2", model2)
        model3 = Model()
        model2.add_submodel("m3", model3)
        model4 = Model()
        model3.add_submodel("m4", model4)
        self.assertIs(model1.submodels["m2", "m3", "m4"], model4)

    def test_model_invalid_operation(self):
        model1 = Model()
        model2 = Model()
        model1.add_submodel("m", model2)
        param = Parameter()
        model1.add_parameter("p", param)
        with self.assertRaises(AttributeError):
            model1.params = _ModelParameter(model1)
        with self.assertRaises(AttributeError):
            model1.submodels = _ModelSubModel(model1)
        with self.assertRaises(AttributeError):
            del model1.params
        with self.assertRaises(AttributeError):
            del model1.submodels
        with self.assertRaises(TypeError):
            del model1.params["p"]
        with self.assertRaises(TypeError):
            del model1.submodels["m"]
