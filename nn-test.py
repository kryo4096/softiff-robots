import taichi as ti
import random

class Optimizer:
    def __init__(self):
        self.learning_rate = 1e-3

    def get_learning_rate(self):
        return self.learning_rate


class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_size, output_size):
        self.input = ti.field(dtype=ti.f32, shape=input_size, needs_grad=True)
        self.input_optimizer = Optimizer()

        self.hidden_layer = ti.field(dtype=ti.f32, shape=(input_size, hidden_layer_size), needs_grad=True)
        self.hidden_weight = ti.field(dtype=ti.f32, shape=(input_size, hidden_layer_size), needs_grad=True)
        self.hidden_bias = ti.field(dtype=ti.f32, shape=input_size, needs_grad=True)
        self.hidden_optimizer = Optimizer()

        self.output = ti.field(dtype=ti.f32, shape=output_size, needs_grad=True)
        self.output_weight = ti.field(dtype=ti.f32, shape=(hidden_layer_size, output_size), needs_grad=True)
        self.output_bias = ti.field(dtype=ti.f32, shape=hidden_layer_size, needs_grad=True)
        self.output_optimizer = Optimizer()

        self.init_variables()

    def init_variables(self):
        for j in range(self.hidden_layer.shape):
            self.hidden_bias = (random.random() - 0.5) * 2 * 10
            for i in range(self.input.shape):
                self.hidden_weight[i][j] = (random.random() - 0.5) * 2 * 10

        for j in range(self.output.shape):
            self.output_bias = (random.random() - 0.5) * 2 * 10
            for i in range(self.hidden_layer.shape):
                self.output_weight[i][j] = (random.random() - 0.5) * 2 * 10

    def compute_output(self, input_vector):
        self.input = input_vector
        self.hidden_layer = self.hidden_weight @ self.input + self.hidden_bias
        self.output = self.output_weight @ self.hidden_layer + self.output_bias
        return self.output

    def update_values(self):
        self.hidden_weight -= self.hidden_optimizer.get_learning_rate() * self.hidden_weight
        self.hidden_bias -= self.hidden_optimizer.get_learning_rate() * self.hidden_bias

        self.output_weight = self.output_optimizer.get_learning_rate() * self.output_weight
        self.output_bias = self.output_optimizer.get_learning_rate() * self.output_bias


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
    def __init__(self, robot, input_size=4, hidden_layer_size=10, output_size = 4):
        self.robot = robot
        self.nn = NeuralNetwork(input_size, hidden_layer_size, output_size)

    def do_control(self):
        pos = self.robot.get_position()
        output = self.nn.compute_output(pos)
        self.robot.apply_actuation(output)

    def update_gradients(self):
        self.nn.update_values()


def main():

    ti.init(arch=ti.gpu)

    robot = Robot()
    controller = RobotController(robot)

    gui = ti.GUI("Robot Simulation")

    while gui.running:
        for i in range(100):
            controller.do_control()
        controller.update_gradients()

        robot.draw(gui)


if __name__ == "__main__":
    main()

