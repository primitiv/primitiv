from primitiv import Trainer, Parameter, Device, Model, Shape
from primitiv import devices as D
from primitiv import initializers as I
from primitiv._model import _ModelParameter
from primitiv._model import _ModelSubModel
from primitiv import tensor_operators as tF

import numpy as np

import unittest
import tempfile


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

    def test_model_load_save(self):
        submodel = TestModel()
        submodel.sp1 = Parameter([2, 4], I.Constant(0))
        submodel.sp1.value = tF.input(np.array([[0,1,2,3],[4,5,6,7]]))
        submodel.sp2 = Parameter([2, 4], I.Constant(0))
        submodel.sp2.value = tF.input(np.array([[9,8,7,6],[5,4,3,2]]))
        submodel.auto_add_attributes()
        parentmodel = TestModel()
        parentmodel.p1 = Parameter([4, 2], I.Constant(0))
        parentmodel.p1.value = tF.input(np.array([[0,1],[2,3],[4,5],[6,7]]))
        parentmodel.p2 = Parameter([4, 2], I.Constant(0))
        parentmodel.p2.value = tF.input(np.array([[9,8],[7,6],[5,4],[3,2]]))
        parentmodel.sub = submodel
        parentmodel.auto_add_attributes()
        submodel_load = TestModel()
        submodel_load.sp1 = Parameter()
        submodel_load.sp2 = Parameter()
        submodel_load.auto_add_attributes()
        parentmodel_load = TestModel()
        parentmodel_load.p1 = Parameter()
        parentmodel_load.p2 = Parameter()
        parentmodel_load.sub = submodel_load
        parentmodel_load.auto_add_attributes()
        with tempfile.NamedTemporaryFile() as fp:
            parentmodel.save(fp.name)
            parentmodel_load.load(fp.name)
        self.assertTrue((parentmodel_load.p1.value.to_ndarrays()[0] == np.array([[0,1],[2,3],[4,5],[6,7]])).all())
        self.assertTrue((parentmodel_load.p2.value.to_ndarrays()[0] == np.array([[9,8],[7,6],[5,4],[3,2]])).all())
        self.assertTrue((parentmodel_load.sub.sp1.value.to_ndarrays()[0] == np.array([[0,1,2,3],[4,5,6,7]])).all())
        self.assertTrue((parentmodel_load.sub.sp2.value.to_ndarrays()[0] == np.array([[9,8,7,6],[5,4,3,2]])).all())

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

    def test_model_get_all_parameters(self):
        submodel = TestModel()
        submodel.sp1 = Parameter()
        submodel.sp2 = Parameter()
        submodel.auto_add_attributes()
        parentmodel = TestModel()
        parentmodel.p1 = Parameter()
        parentmodel.p2 = Parameter()
        parentmodel.sub = submodel
        parentmodel.auto_add_attributes()
        params = parentmodel.get_all_parameters()
        self.assertIs(params[("p1",)], parentmodel.p1)
        self.assertIs(params[("p2",)], parentmodel.p2)
        self.assertIs(params[("sub", "sp1")], parentmodel.sub.sp1)
        self.assertIs(params[("sub", "sp2")], parentmodel.sub.sp2)
