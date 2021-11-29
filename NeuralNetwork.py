import taichi as ti
import numpy as np

import random
import time

ti.set_logging_level(ti.ERROR)
ti.init(arch=ti.gpu)


energy = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

gui = ti.GUI("Ball Throwing  Simulation")

SHOULD_DRAW = True


@ti.func
def relu(value):
    return max(0, value)


@ti.kernel
def reset_energy():
    energy[None] = 0


@ti.kernel
def calculate_initial_loss(v_x_init: ti.template(), v_y_init: ti.template()):
    pass
    #energy[None] += relu(v_x_init[None] - 1)
    #energy[None] += relu(-v_y_init[None])


@ti.kernel
def calculate_loss(true_x: ti.f32, true_y: ti.f32, x_final: ti.template(), y_final: ti.template()):
    energy[None] += (x_final[None] - true_x)**2 + (y_final[None] - true_y)**2
    #energy[None] -= 0.2*y_final[None]


@ti.data_oriented
class Optimizer:
    def __init__(self):
        self.learning_rate = 1e-3
        self.t = 0

    def get_learning_rate(self):
        self.t += 1
        return self.learning_rate * (1 + pow(100, 1-self.t/100))


@ti.data_oriented
class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_size1, hidden_layer_size2, output_size):
        self.input_size = input_size

        self.hidden_layer_size1 = hidden_layer_size1
        self.hidden_weight1 = ti.Matrix.field(n=hidden_layer_size1, m=input_size, dtype=ti.f32, shape=(), needs_grad=True)
        self.hidden_bias1 = ti.Vector.field(n=hidden_layer_size1, dtype=ti.f32, shape=(), needs_grad=True)
        self.hidden_optimizer1 = Optimizer()

        self.hidden_layer_size2 = hidden_layer_size2
        self.hidden_weight2 = ti.Matrix.field(n=hidden_layer_size2, m=hidden_layer_size1, dtype=ti.f32, shape=(), needs_grad=True)
        self.hidden_bias2 = ti.Vector.field(n=hidden_layer_size2, dtype=ti.f32, shape=(), needs_grad=True)
        self.hidden_optimizer2 = Optimizer()

        self.output_size = output_size
        self.output = ti.Vector.field(n=output_size, dtype=ti.f32, shape=(), needs_grad=True)
        self.output_weight = ti.Matrix.field(n=output_size, m=hidden_layer_size2, dtype=ti.f32, shape=(), needs_grad=True)
        self.output_bias = ti.Vector.field(n=output_size, dtype=ti.f32, shape=(), needs_grad=True)
        self.output_optimizer = Optimizer()

        self.init_variables()

    def init_variables(self):
        for j in range(self.hidden_layer_size1):
            self.hidden_bias1[None][j] = (random.random() - 0.5) * 2 * 1
            for i in range(self.input_size):
                self.hidden_weight1[None][j, i] = (random.random() - 0.5) * 2 * 1

        for j in range(self.hidden_layer_size2):
            self.hidden_bias2[None][j] = (random.random() - 0.5) * 2 * 1
            for i in range(self.hidden_layer_size1):
                self.hidden_weight2[None][j, i] = (random.random() - 0.5) * 2 * 1

        for j in range(self.output_size):
            self.output_bias[None][j] = (random.random() - 0.5) * 2 * 1
            for i in range(self.hidden_layer_size2):
                self.output_weight[None][j, i] = (random.random() - 0.5) * 2 * 1

    @ti.kernel
    def compute_output(self, x: ti.f32, y: ti.f32):
        input_vector = ti.Vector([x, y])
        hidden_layer1 = self.hidden_weight1[None] @ input_vector + self.hidden_bias1[None]
        hidden_layer2 = self.hidden_weight2[None] @ hidden_layer1 + self.hidden_bias2[None]
        self.output[None] = self.output_weight[None] @ hidden_layer2 + self.output_bias[None]

    @ti.kernel
    def update_values(self):
        self.hidden_weight1[None] -= self.hidden_optimizer1.get_learning_rate() * self.hidden_weight1.grad[None]
        self.hidden_bias1[None] -= self.hidden_optimizer1.get_learning_rate() * self.hidden_bias1.grad[None]

        self.hidden_weight2[None] -= self.hidden_optimizer2.get_learning_rate() * self.hidden_weight2.grad[None]
        self.hidden_bias2[None] -= self.hidden_optimizer2.get_learning_rate() * self.hidden_bias2.grad[None]

        self.output_weight[None] -= self.output_optimizer.get_learning_rate() * self.output_weight.grad[None]
        self.output_bias[None] -= self.output_optimizer.get_learning_rate() * self.output_bias.grad[None]


@ti.data_oriented
class Simulation():
    def __init__(self):
        self.x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.y = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.v_x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.v_y = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        self.x_init = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.y_init = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.v_x_init = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.v_y_init = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        self.ground_spring_constant = 1e6
        self.mass = 1
        self.g = -9.81
        self.dt = 3e-4
        #self.gui = ti.GUI("Simulation_Visualization")

    @ti.kernel
    def init(self, x: ti.f32, y: ti.f32, v_x: ti.template(), v_y: ti.template()):
        self.x[None] = x
        self.y[None] = y
        self.v_x[None] = v_x[None][0]
        self.v_y[None] = v_y[None][1]

        self.x_init[None] = x
        self.y_init[None] = y
        self.v_x_init[None] = v_x[None][0]
        self.v_y_init[None] = v_y[None][1]

    @ti.func
    def ground_force(self) -> ti.f32:
        return self.ground_spring_constant * relu(-self.y[None])

    @ti.func
    def gravity(self) -> ti.f32:
        return self.mass * self.g

    @ti.func
    def x_friction(self) -> ti.f32:
        return -self.v_x[None] * 1e6 *relu(-self.y[None])

    @ti.func
    def y_friction(self) -> ti.f32:
        return -self.v_y[None] * 1e6 *relu(-self.y[None])

    @ti.kernel
    def explicit_euler(self):
        v_pre = self.v_y[None]
        self.v_y[None] += (self.ground_force() + self.gravity() + self.y_friction()) * self.dt
        self.y[None] += v_pre * self.dt
        self.x[None] += self.v_x[None] * self.dt

    @ti.kernel
    def symplectic_euler(self):
        self.v_x[None] += (self.x_friction()) * self.dt
        self.v_y[None] += (self.ground_force() + self.gravity() + self.y_friction()) * self.dt
        self.y[None] += self.v_y[None] * self.dt
        self.x[None] += self.v_x[None] * self.dt

def generate_data(current_iter, max_iter):
    if current_iter < max_iter:
        return random.random(), random.random()
    else:
        return random.random(), random.random()

def main():
    iter_count = 0

    sim = Simulation()
    nn = NeuralNetwork(2, 10, 10, 2)

    training_iterations = 10000

    while gui.running:
        x, y = generate_data(iter_count, training_iterations)
        with ti.Tape(loss=energy):
            nn.compute_output(x, y)
            sim.init(0.1, 0.1, nn.output, nn.output)
            calculate_initial_loss(sim.v_x_init, sim.v_y_init)
            for i in range(1000):

                sim.symplectic_euler()

                if SHOULD_DRAW and i % 30 == 0 and iter_count % 10 == 0: #iter_count > training_iterations and
                    time.sleep(0.033)
                    gui.circle(pos=(x, y), radius=10, color=0xFF0000)
                    gui.circle(pos=(sim.x[None], sim.y[None]), radius=10)
                    gui.show()

            calculate_loss(x, y, sim.x, sim.y)

            print(f"Completed iteration {iter_count} with values x:{sim.x[None]}, v_x_init: {sim.v_x_init[None]}, v_y_init: {sim.v_y_init[None]} with final loss {energy[None]}")

        if iter_count < training_iterations:
            nn.update_values()
        iter_count += 1


if __name__ == "__main__":
    main()
