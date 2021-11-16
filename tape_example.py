import taichi as ti
import random

import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)


x_size = 4
x = ti.field(dtype=ti.f32, shape=x_size, needs_grad=True)
output = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

target = 5

learning_rate = 1e-3


def init():
    for i in range(x_size):
        x[i] = (random.random() - 0.5) * 10
    loss[None] = 1


def clear_output():
    output[None] = 0


@ti.kernel
def get_output():
    for i in range(x_size):
        output[None] += x[i]


@ti.kernel
def calculate_loss():
    loss[None] = ti.abs(output[None] - target)


def do_iter():
    get_output()
    calculate_loss()


def update_var():
    for i in range(x_size):
        x[i] -= learning_rate * x.grad[i]


NUM_ITER = 0
ITERATIONS = []
T = []
OUTPUT = []

init()
while loss[None] > 0.01:
    clear_output()

    with ti.Tape(loss):
        do_iter()

    if loss[None] > 0.01:
        update_var()

    ITERATIONS.append(NUM_ITER)
    T.append(target)
    OUTPUT.append(output[None])

    NUM_ITER += 1
    print("Loss is", loss[None])


print("Final Values are", x[0], x[1], x[2], x[3], "and sum of", x[0] + x[1] + x[2] + x[3], "and target is", target)

plt.plot(ITERATIONS, T, label="target")
plt.plot(ITERATIONS, OUTPUT, color="r", label="output network")

plt.legend()
plt.show()
