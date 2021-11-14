import taichi as ti
import random

import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)


x_size = 4
x = ti.field(dtype=ti.f32, shape=x_size, needs_grad=True)
output = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

x_2 = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
output_2 = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
loss_2 = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

target = 5

learning_rate = 1e-3
learning_rate_2 = 1e-3


def init():
    for i in range(x_size):
        x[i] = random.random()
    x_2[None] = random.random()
    loss[None] = 1
    loss_2[None] = 1

def clear_output():
    output[None] = 0


def clear_output_2():
    output_2[None] = 0


@ti.kernel
def get_output():
    for i in range(x_size):
        output[None] += x[i]


@ti.kernel
def get_output_2():
    output_2[None] = x_2[None]


@ti.kernel
def calculate_loss():
    loss[None] = ti.abs(output[None] - output_2[None])


@ti.kernel
def calculate_loss_2():
    loss_2[None] = ti.abs(output_2[None] - target)


def do_iter_2():
    get_output_2()
    calculate_loss_2()


def update_var_2():
    x_2[None] -= learning_rate_2 * x_2.grad[None]


def main_loop_2():
    clear_output_2()
    with ti.Tape(loss_2):
        do_iter_2()
    if loss_2[None] > 0.01: update_var_2()


def do_iter(loss_value):
    rand = random.random()
    if loss_2[None] > 0.01 and rand > loss_value:
        main_loop_2()

    get_output()
    calculate_loss()


def update_var():
    for i in range(x_size):
        x[i] -= learning_rate * x.grad[i]


num_iter = 0
iter = []
t = []
o_1 = []
o_2 = []

init()
while (loss[None] > 0.01) or (loss_2[None] > 0.01):
    clear_output()
    l = loss[None]
    with ti.Tape(loss):
        do_iter(l)
    if loss[None] > 0.01: update_var()


    iter.append(num_iter)
    t.append(target)
    o_1.append(output[None])
    o_2.append(output_2[None])

    num_iter += 1
    print("Values of first NN is", x[0] + x[1] + x[2] + x[3], \
      "\nValues of second NN are", x_2[None], "and target is", target, \
      "\nLosses are", loss[None], "and", loss_2[None])


print("Final Values are", x[0], x[1], x[2], x[3], "and sum of", x[0] + x[1] + x[2] + x[3], \
      "Final Values of second NN are", x_2[None], "and target is", target)

plt.plot(iter, t, label="target")
plt.plot(iter, o_1, color="r", label="output outer network")
plt.plot(iter, o_2, color="g", label="output inner network")

plt.legend()
plt.show()
