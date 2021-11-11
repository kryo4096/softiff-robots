import taichi as ti
import random


@ti.data_oriented
class NeuralNet:
    def __init__(self):
        self.input_size = 2
        self.input = ti.field(dtype=ti.f32, shape=input_size, need_grad = true)
        self.weight = ti.field(dtype=ti.f32, shape=input_size, needs_grad = true)
        self.bias = ti.field(dtype=ti.f32, shape=input_size, needs_grad = true)



        self.initialize()


    def initialize(self):
        for i in range(self.input_size
            self.weight[i] = random.random()*100 - 50
            self.bias[i] = random.random()*100 - 50


    def generate_prediction(self, v_x, y_0):
        return x_0*self.weight[0] + self.bias[0], y_0*self.weight[1] + self.bias[1]


    def do_iteration(self):
        v_x, v_y = self.generate_prediction(0, 0)


    def dump_parameters(self):
        print("Weights:")
        for i in range(self.input_size):
            print(self.weight[i])
        print("Biases:")
        for i in range(self.input_size):
            print(self.bias[i])



@ti.data_oriented
class Simulator:
    def __init__(self):
        pass

    @ti.pyfunc
    def create_simulation(self, x_0, y_0, v_x, v_y):
        vel = ti.sqrt(v_x**2 + v_y**2)
        d = vel2*ti.sin(2*ti.atan(v_y/v_x))/9.81
        return v_x/ti.abs(v_x) * d


def main():
    ti.init(arch=ti.gpu)
    gui = ti.GUI('Autodiff example')

    nn = NeuralNet()
    sim = Simulator()

    for i in range(1000):
        x_0, y_0 = random.random(), random.random()
        nn.do_iteration()


    nn.dump_parameters()



if __name__ == "__main__":
    main()
