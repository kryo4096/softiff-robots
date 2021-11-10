import taichi as ti
import random

ti.init(arch=ti.gpu)

'''
    ARCHITECTURE:
    REWARD: x_value
    INPUT: Size 2, x_0, y_0

    Layer 1: Size 2, Inputs: 2 from INPUT, OUTPUT

    OUTPUT: Size 2, v_x, v_y

    INPUT       LAYER 1       OUTPUT
    x_0 ------>  N_1 ---------> v_y
       \   /
        \ /
        /\
    y_0--------> N_2 ---------> v_y
    '''

@ti.data_oriented
class NeuralNetwork:

    def __init__(self):
        self.input_size = 2
        self.layer1_size = 2
        self.output_size = 2

        self.inputs = ti.field(dtype=ti.f32, shape=self.input_size)
        self.layer1 = ti.field(dtype=ti.f32, shape=self.layer1_size)
        self.output = ti.field(dtype=ti.f32, shape=self.output_size)

        self.weights_layer1 = ti.Matrix.field(self.input_size, self.layer1_size, dtype=ti.f32, shape=())
        self.weights_output = ti.Matrix.field(self.layer1_size, self.output_size, dtype=ti.f32, shape=())

    @ti.kernel
    def init(self, x_init: ti.f32, y_init: ti.f32):
        self.inputs[0] = x_init
        self.inputs[1] = y_init

        for i, j in ti.ndrange(self.input_size, self.layer1_size):
            self.weights_layer1[i, j] = ti.random()

        for i, j in ti.ndrange(self.layer1_size, self.output_size):
            self.weights_output[i, j] = ti.random()

    def get_x_vel(self):
        return self.output[0]

    def get_y_vel(self):
        return self.output[1]

    def train(self):
        pass

    def eval(self):
        self.layer1 = self.weights_layer1.transpose() @ self.inputs
        self.output = self.weights_output.transpose() @ self.layer1


@ti.data_oriented
class Simulation:
    def __init__(self):

        self.dt = 1e-4

        self.x = ti.Vector.field(2, ti.f32, 1, needs_grad=True)  # position of particles
        self.v = ti.Vector.field(2, ti.f32, 1)  # velocity of particles
        self.U = ti.field(float, (), needs_grad=True)  # potential energy

        self.m = 1
        self.g = 9.81

    @ti.kernel
    def compute_U(self):
        h = self.x[0][1]
        self.U[None] += self.m * self.g * h

    @ti.kernel
    def advance(self):
        self.v[0] += self.dt * -self.x.grad[0]
        self.x[0] += self.dt * self.v[0]

    def substep(self):
        with ti.Tape(self.U):
            self.compute_U()
        self.advance()

    @ti.kernel
    def init(self, x: ti.f32, y: ti.f32, v_x: ti.f32, v_y: ti.f32):
        self.x[0] = [x, y]
        self.v[0] = [v_x, v_y]

gui = ti.GUI('Autodiff gravity')
nn = NeuralNetwork()

sim = Simulation()


while gui.running:

    x_init, y_init = random.random(), random.random()
    nn.init(x_init, y_init)

    nn.eval()

    x_v_init, y_v_init = nn.get_x_vel(), nn.get_y_vel()


    sim.init(x_init, y_init, x_v_init, y_v_init)


    for i in range(100):
        sim.substep()
        if sim.x[0][1] < 0:
            print("Final Distance achieved: ", sim.x[0][0])
            gui.running = False
            break

    gui.circles(sim.x.to_numpy(), radius=8.0)
    gui.show()
