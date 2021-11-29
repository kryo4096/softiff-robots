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
def calculate_loss(p_angle: ti.template(), p_length: ti.template()):
    energy[None] -= ti.cos(p_angle[None]) * p_length[None]
    #energy[None] -= 0.2*y_final[None]


@ti.data_oriented
class Optimizer:
    def __init__(self):
        self.learning_rate = 1e-7
        self.t = 0

    def get_learning_rate(self):
        self.t += 1
        return self.learning_rate * (1 + pow(10, 1-self.t/100))


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
    def compute_output(self, c_x: ti.template(), c_v: ti.template(), c_a: ti.template(), p_a: ti.template(), p_v: ti.template(), p_acc: ti.template()):
        input_vector = ti.Vector([c_x[None], c_v[None], c_a[None], p_a[None], p_v[None], p_acc[None]])
        hidden_layer1 = self.hidden_weight1[None] @ input_vector + self.hidden_bias1[None]
        hidden_layer2 = self.hidden_weight2[None] @ hidden_layer1 + self.hidden_bias2[None]
        self.output[None] = self.output_weight[None] @ hidden_layer2 + self.output_bias[None]
        self.output[None] *= 0

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
        self.cart_x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.cart_y = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.cart_v_x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.cart_a_x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        self.cart_x_control = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        self.pendulum_angle = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.pendulum_v_angle = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.pendulum_a_angle = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.pendulum_length = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        self.border_spring_constant = 1e4
        self.car_mass = 10
        self.pendulum_mass = 1
        self.g = -9.81
        self.dt = 3e-4
        #self.gui = ti.GUI("Simulation_Visualization")

    @ti.kernel
    def init(self, c_x: ti.f32, c_v: ti.f32, c_a: ti.f32, p_a: ti.f32, p_v: ti.f32, p_acc: ti.f32):
        self.cart_x[None] = c_x
        self.cart_y[None] = 0.5
        self.cart_v_x[None] = c_v
        self.cart_a_x[None] = c_a

        self.cart_x_control[None] = 0

        self.pendulum_angle[None] = p_a
        self.pendulum_v_angle[None] = p_v
        self.pendulum_a_angle[None] = p_acc
        self.pendulum_length[None] = 0.1

    @ti.kernel
    def update(self, c_c: ti.template()):
        self.cart_x_control[None] = c_c[None][0]


    @ti.func
    def side_force_cart(self) -> ti.f32:
        force_r = self.border_spring_constant * (relu(self.pendulum_length[None]-self.cart_x[None]))
        force_l = self.border_spring_constant * (-relu(self.cart_x[None]-self.pendulum_length[None]-1))
        return force_r + force_l + self.cart_x_control[None]

    @ti.func
    def pendulum_gravity(self) -> ti.f32:
        return self.pendulum_mass * self.g

    @ti.kernel
    def symplectic_euler(self):
        ml2 = 0.5 * self.pendulum_mass * self.pendulum_length[None]
        x_acc = self.car_mass + self.pendulum_mass + ml2*ti.cos(self.pendulum_angle[None])
        angle_acc = ml2*ti.cos(self.pendulum_angle[None]) + ml2*0.5*self.pendulum_length[None] + ml2/6*self.pendulum_length[None]
        angle_vel = -ml2*ti.sin(self.pendulum_angle[None])
        gravity = ml2*self.g*ti.sin(self.pendulum_angle[None])

        self.cart_a_x[None] += (self.pendulum_a_angle[None]*angle_acc + self.pendulum_v_angle[None] * angle_vel + gravity + self.side_force_cart()) / x_acc * self.dt
        self.pendulum_a_angle[None] += (self.cart_a_x[None]*x_acc + self.pendulum_v_angle[None] * angle_vel + gravity + self.side_force_cart()) / angle_acc * self.dt

        self.cart_v_x[None] += self.cart_a_x[None] * self.dt
        self.pendulum_v_angle[None] += self.pendulum_a_angle[None] * self.dt

        self.cart_x[None] += self.cart_v_x[None] * self.dt
        self.pendulum_angle[None] += self.pendulum_v_angle[None] * self.dt


def generate_data(current_iter, max_iter):
    #c_x, c_v, c_a, p_a, p_v, p_acc
    return 0.5, 0, 0, 0.1, 0, 0
    if current_iter < max_iter:
        return random.random()*0.8 + 0.1, random.random()-0.5, random.random()-0.5, random.random()*2*3.14, random.random()-0.5, random.random()-0.5
    else:
        return random.random()*0.8 + 0.1, random.random()-0.5, random.random()-0.5, random.random()*2*3.14, random.random()-0.5, random.random()-0.5

def main():
    iter_count = 0

    sim = Simulation()
    nn = NeuralNetwork(6, 5, 5, 1)

    training_iterations = 1000

    while gui.running:
        c_x, c_v, c_a, p_a, p_v, p_acc = generate_data(iter_count, training_iterations)
        sim.init(c_x, c_v, c_a, p_a, p_v, p_acc)
        with ti.Tape(loss=energy):
            for i in range(10000):
                nn.compute_output(sim.cart_x, sim.cart_v_x, sim.cart_a_x, sim.pendulum_angle, sim.pendulum_v_angle, sim.pendulum_a_angle)
                sim.update(nn.output)
                sim.symplectic_euler()

                if SHOULD_DRAW and i % 30 == 0: #iter_count > training_iterations and
                    time.sleep(0.033)
                    gui.circle(pos=(sim.cart_x[None], sim.cart_y[None]), radius=20, color=0xFF0000)
                    gui.circle(pos=(sim.cart_x[None] + ti.sin(sim.pendulum_angle[None])*sim.pendulum_length[None], 0.5+ti.cos(sim.pendulum_angle[None]) * sim.pendulum_length[None]), radius=10)
                    gui.show()



            calculate_loss(sim.pendulum_angle, sim.pendulum_length)

            print(f"Completed iteration {iter_count} with cart_x {sim.cart_x[None]} cart_y {sim.cart_y[None]} and final loss {energy[None]}")

        if iter_count < training_iterations:
            nn.update_values()
        iter_count += 1


if __name__ == "__main__":
    main()
