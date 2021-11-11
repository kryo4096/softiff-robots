import taichi as ti
import random



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

        self.weights_layer1 = ti.field(dtype=ti.f32, shape=(self.input_size, self.layer1_size), needs_grad=True)
        self.weights_output = ti.field(dtype=ti.f32, shape=(self.layer1_size, self.output_size), needs_grad=True)

        self.sim = Simulation()

        self.performance = ti.field(ti.f32, (), needs_grad=True)

    @ti.kernel
    def init(self, x_init: ti.f32, y_init: ti.f32):
        self.inputs[0] = x_init
        self.inputs[1] = y_init

        for i, j in ti.ndrange(self.input_size, self.layer1_size):
            self.weights_layer1[i, j] = ti.random()

        for i, j in ti.ndrange(self.layer1_size, self.output_size):
            self.weights_output[i, j] = ti.random()

    @ti.kernel
    def update_perfomance(self):
        self.calculate_loss(self.sim.x[0][0])
        self.performance[None] = self.loss

        self.update_weights()
        self.eval()



    @ti.func
    def update_weights(self):


        for i, j in ti.static(ti.ndrange(self.input_size, self.layer1_size)):
            self.weights_layer1[i, j] += self.weights_layer1.grad[i, j]


        for i, j in ti.static(ti.ndrange(self.layer1_size, self.output_size)):
            self.weights_output[i, j] += self.weights_output.grad[i, j]


    def calculate_loss(self, value):
        self.loss = -value


    def train(self):
        self.loss = 0
        while True:
            self.sim.substep()
            if self.sim.x[0][1] < 0:
                #print("Final Distance achieved: ", self.sim.x[0][0])
                break

        self.update_perfomance()
        self.update_perfomance.grad()


    def get_velocities(self):
        self.eval()
        return self.output[0], self.output[1]

    @ti.pyfunc
    def eval(self):
        for j in ti.static(range(self.layer1_size)):
            self.layer1[j] = 0

        for j in ti.static(range(self.output_size)):
            self.output[j] = 0


        for i, j in ti.static(ti.ndrange(self.input_size, self.layer1_size)):
            self.layer1[j] += self.weights_layer1[i, j] * self.inputs[i]

        for i, j in ti.static(ti.ndrange(self.layer1_size, self.output_size)):
            self.output[j] += self.weights_output[i, j] * self.layer1[i]



@ti.data_oriented
class Simulation:
    def __init__(self):

        self.dt = 1e-4

        self.x = ti.Vector.field(2, ti.f32, 1, needs_grad=True)  # position of particles
        self.v = ti.Vector.field(2, ti.f32, 1)  # velocity of particles
        self.U = ti.field(ti.f32, (), needs_grad=True)  # potential energy

        self.m = 1
        self.g = 9.81

    @ti.kernel
    def init(self, x: ti.f32, y: ti.f32, v_x: ti.f32, v_y: ti.f32):
        self.x[0] = [x, y]
        self.v[0] = [v_x, v_y]

        print("Initialized Simulation with p_init ", x, y, " and v_init ", v_x, v_y)

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


ti.init(arch=ti.gpu)
gui = ti.GUI('Autodiff gravity')
nn = NeuralNetwork()



while gui.running:

    x_init, y_init = random.random(), random.random()

    nn.init(x_init, y_init)



    for i in range(1000):
        x_v_init, y_v_init = nn.get_velocities()
        nn.sim.init(x_init, y_init, x_v_init, y_v_init)
        nn.train()

    gui.circles(nn.sim.x.to_numpy(), radius=8.0)
    gui.show()
