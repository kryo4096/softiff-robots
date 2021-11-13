import taichi as ti
import random
import math


@ti.data_oriented
class NeuralNet:
    #inputs are x, y, v_x, v_y
    def __init__(self):
        self.input_size = 4

        self.input = ti.field(dtype=ti.f32, shape=self.input_size, needs_grad = True)
        self.weight = ti.field(dtype=ti.f32, shape=self.input_size, needs_grad = True)
        self.bias = ti.field(dtype=ti.f32, shape=self.input_size, needs_grad = True)
        self.output = ti.field(dtype=ti.f32, shape=self.input_size)

        self.loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        self.sim = Simulator()

        #self.gui = ti.GUI("Simulation")

        self.initialize()

    def initialize(self):
        for i in range(self.input_size):
            self.input[i] = random.random()*10 - 5
            self.weight[i] = random.random()*100 - 50
            self.bias[i] = random.random()*100 - 50

    def generate_prediction(self):
        for i in range(self.input_size):
            self.output[i] = self.input[i]*self.weight[i] + self.bias[i]

        self.loss[None] = self.sim.run(self.output[0], self.output[1], self.output[2], self.output[3]) #input_size
        print("Loss was ", self.loss[None])


    def update_network(self):
        for i in range(self.input_size):
            self.input[i] += self.input.grad[i]
            self.weight[i] += self.weight.grad[i]
            self.bias[i] += self.bias.grad[i]


    def do_iteration(self):
        with ti.Tape(self.loss):
            self.generate_prediction()
        self.update_network()


    def dump_parameters(self):
        print("Weights:")
        for i in range(self.input_size):
            print(self.weight[i])
        print("Weight gradients:")
        for i in range(self.input_size):
            print(self.weight.grad[i])
        print("Biases:")
        for i in range(self.input_size):
            print(self.bias[i])
        print("Biasgradients:")
        for i in range(self.input_size):
            print(self.bias.grad[i])



@ti.data_oriented
class Simulator:
    def __init__(self):
        self.delta_t = 1e-5
        self.g = -9.81

    def run(self, x_0: ti.f32, y_0: ti.f32, v_x: ti.f32, v_y: ti.f32):
        x, y = x_0, y_0
        if x < 0: y = -1
        if y < 0: x = y
        if math.sqrt(v_x**2 + v_y**2) > 1:
            y = -1
            x = -math.sqrt(v_x**2 + v_y**2) + 1
        #iter_count = 0
        while y > 0:
            v_y += self.g * self.delta_t
            x += v_x * self.delta_t
            y += v_y * self.delta_t
            '''
            if iter_count % 30 == 0:
                x_pos: float = x
                y_pos: float = y
                pos = np.array([x_pos, y_pos])
                gui.circle(pos, color=0xFF0000)
            ++iter_count
            '''
        return x
7

def main():
    ti.init(arch=ti.gpu)
    #gui = ti.GUI('Autodiff example')

    nn = NeuralNet()

    for i in range(10):
        nn.do_iteration()
        nn.dump_parameters()



if __name__ == "__main__":
    main()
