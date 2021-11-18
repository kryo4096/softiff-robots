import random
import time

import taichi as ti
import numpy as np



ti.init(arch=ti.gpu)

energy = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

gui = ti.GUI("Robot Simulation")

SHOULD_DRAW = False


@ti.func
def relu(value):
    return max(0, value)


@ti.kernel
def reset_energy():
    energy[None] = 0


@ti.kernel
def calculate_initial_loss(v_x_init: ti.f32, v_y_init: ti.f32):
    energy[None] += 10 * relu(v_x_init - 1)
    energy[None] += 10 * relu(v_y_init - 1)


@ti.kernel
def calculate_loss(x_final: ti.f32):
    energy[None] -= x_final


@ti.data_oriented
class Optimizer:
    def __init__(self):
        self.learning_rate = 1e-3
        self.t = 0

    def get_learning_rate(self):
        self.t += 1
        return self.learning_rate + pow(2, -self.t / 10)


@ti.data_oriented
class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_size, output_size):
        self.input_size = input_size

        self.hidden_layer_size = hidden_layer_size
        self.hidden_weight = ti.Matrix.field(n=hidden_layer_size, m=input_size, dtype=ti.f32, shape=(), needs_grad=True)
        self.hidden_bias = ti.Vector.field(n=hidden_layer_size, dtype=ti.f32, shape=(), needs_grad=True)
        self.hidden_optimizer = Optimizer()

        self.output_size = output_size
        self.output = ti.Vector.field(n=output_size, dtype=ti.f32, shape=(), needs_grad=True)
        self.output_weight = ti.Matrix.field(n=output_size, m=hidden_layer_size, dtype=ti.f32, shape=(), needs_grad=True)
        self.output_bias = ti.Vector.field(n=output_size, dtype=ti.f32, shape=(), needs_grad=True)
        self.output_optimizer = Optimizer()

        self.init_variables()

    def init_variables(self):
        for j in range(self.hidden_layer_size):
            self.hidden_bias[None][j] = (random.random() - 0.5) * 2 * 10
            for i in range(self.input_size):
                self.hidden_weight[None][j, i] = (random.random() - 0.5) * 2 * 10

        for j in range(self.output_size):
            self.output_bias[None][j] = (random.random() - 0.5) * 2 * 10
            for i in range(self.hidden_layer_size):
                self.output_weight[None][j, i] = (random.random() - 0.5) * 2 * 10

    @ti.kernel
    def compute_output(self, x: ti.f32, y: ti.f32):
        input = ti.Vector([x, y])
        hidden_layer = self.hidden_weight[None] @ input + self.hidden_bias[None]
        self.output[None] = self.output_weight[None] @ hidden_layer + self.output_bias[None]

    @ti.kernel
    def update_values(self):
        print("Norms of vectors are", self.hidden_weight.grad[None].norm(), self.hidden_bias.grad[None].norm(), self.output_weight.grad[None].norm(), self.output_bias.grad[None].norm())

        self.hidden_weight[None] -= self.hidden_optimizer.get_learning_rate() * self.hidden_weight.grad[None]
        self.hidden_bias[None] -= self.hidden_optimizer.get_learning_rate() * self.hidden_bias.grad[None]

        self.output_weight[None] -= self.output_optimizer.get_learning_rate() * self.output_weight.grad[None]
        self.output_bias[None] -= self.output_optimizer.get_learning_rate() * self.output_bias.grad[None]


'''
class Robot:
    def __init__(self):
        pass

    def get_position(self):
        pass

    def apply_actuation(self):
        pass

    def draw(self):
        pass


class RobotController:
    def __init__(self, robot, input_size=2, hidden_layer_size=10, output_size=2):
        self.robot = robot
        self.nn = NeuralNetwork(input_size, hidden_layer_size, output_size)

    def do_control(self):
        pos = self.robot.get_position()
        output = self.nn.compute_output(pos)
        self.robot.apply_actuation(output)

    def update_gradients(self):
        self.nn.update_values()
'''


@ti.data_oriented
class Simulation():
    def __init__(self, x, y, v_x, v_y):
        self.x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.y = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.v_x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.v_y = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        self.x_init = x
        self.y_init = y
        self.v_x_init = v_x
        self.v_y_init = v_y

        self.ground_spring_constant = 1e6
        self.mass = 1
        self.g = -9.81
        self.dt = 1e-3
        #self.gui = ti.GUI("Simulation_Visualization")

        self.init(x, y, v_x, v_y)

    def init(self, x, y, v_x, v_y):
        self.x[None] = x
        self.y[None] = y
        self.v_x[None] = v_x
        self.v_y[None] = v_y

    @ti.kernel
    def explicit_euler(self):
        v_pre = self.v_y[None]
        self.v_y[None] += (self.ground_spring_constant * relu(-self.y[None]) + self.mass * self.g) * self.dt
        self.y[None] += v_pre * self.dt
        self.x[None] += self.v_x[None] * self.dt

    @ti.kernel
    def symplectic_euler(self):
        self.v_y[None] += (self.ground_spring_constant * relu(-self.y[None]) + self.mass * self.g) * self.dt
        self.y[None] += self.v_y[None] * self.dt
        self.x[None] += self.v_x[None] * self.dt


nn = NeuralNetwork(2, 10, 2)

def main():

    while gui.running:

        x, y = 0.1, 0.1

        with ti.Tape(loss=energy):

            nn.compute_output(x, y)
            sim = Simulation(x, y, nn.output[None][0], nn.output[None][1])
            calculate_initial_loss(sim.v_x[None], sim.v_y[None])
            for i in range(800):

                sim.explicit_euler()  # sim.symplectic_euler()

                if SHOULD_DRAW and i % 10 == 0:
                    time.sleep(0.033)
                    gui.circle(pos=(sim.x[None], sim.y[None]), radius=10)
                    gui.show()

            calculate_loss(sim.x[None])
            print(f"Completed iteration with values x:{sim.x[None]}, v_x_init: {sim.v_x_init}, v_y_init: {sim.v_y_init} with final loss {energy[None]}")

        nn.update_values()


if __name__ == "__main__":
    main()
