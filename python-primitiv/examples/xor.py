from primitiv import DefaultScopeDevice
from primitiv import DefaultScopeGraph
from primitiv import Parameter, Shape, Graph
from primitiv import Device, CPUDevice
from primitiv import operators as F
from primitiv import initializers as I
from primitiv import trainers as T

import numpy as np


def main():

    with DefaultScopeDevice(CPUDevice()):
        pw1 = Parameter("w1", [8, 2], I.XavierUniform())
        pb1 = Parameter("b1", [8], I.Constant(0))
        pw2 = Parameter("w2", [1, 8], I.XavierUniform())
        pb2 = Parameter("b2", [], I.Constant(0))

        trainer = T.SGD(0.1)

        trainer.add_parameter(pw1)
        trainer.add_parameter(pb1)
        trainer.add_parameter(pw2)
        trainer.add_parameter(pb2)

        input_data = [
            np.array([ 1,  1], dtype=np.float32),  # Sample 1
            np.array([ 1, -1], dtype=np.float32),  # Sample 2
            np.array([-1,  1], dtype=np.float32),  # Sample 3
            np.array([-1, -1], dtype=np.float32),  # Sample 4
        ]

        output_data = np.array([
             1,  # Label 1
            -1,  # Label 2
            -1,  # Label 3
             1,  # Label 4
        ], dtype=np.float32)

        for i in range(100):
            g = Graph()
            with DefaultScopeGraph(g):
                # Builds a computation graph.
                #x = F.input(shape=Shape([2], 4), data=input_data)
                x = F.input(data=input_data)
                w1 = F.input(param=pw1)
                b1 = F.input(param=pb1)
                w2 = F.input(param=pw2)
                b2 = F.input(param=pb2)
                h = F.tanh(F.matmul(w1, x) + b1)
                y = F.matmul(w2, h) + b2

                # Calculates values.
                y_val = g.forward(y).to_list()
                print("epoch ", i, ":")
                for j in range(4):
                    print("  [", j, "]: ", y_val[j])
                    #t = F.input(shape=Shape([], 4), data=output_data)
                    t = F.input(data=output_data)
                diff = t - y
                loss = F.batch.mean(diff * diff)
                loss_val = g.forward(loss).to_list()[0]
                print("  loss: ", loss_val)
                trainer.reset_gradients()
                g.backward(loss)
                trainer.update()


if __name__ == "__main__":
    main()
