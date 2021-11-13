import taichi as ti
import math
import random

ti.init(arch=ti.gpu)

x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
weight = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
bias = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

learning_rate = 1e-3


def init():
    x[None] = random.random() * 100 - 50
    weight[None] = random.random() * 100 - 50
    bias[None] = random.random() * 100 - 50
    loss[None] = 1

@ti.func
def calculate_loss(value):
    loss[None] = abs(value - 5)

@ti.kernel
def do_iter():
    output = x[None]*weight[None] + bias[None]
    calculate_loss(output)


def update_variables():
    x[None] -= learning_rate * x.grad[None]
    weight[None] -= learning_rate * weight.grad[None]
    bias[None] -= learning_rate * bias.grad[None]

init()

print("Initial x is: ", x[None], "Initial weight is: ", weight[None], "Initial bias is: ", bias[None], "Initial result is", x[None]*weight[None] + bias[None])

num_iter = 0
while True:
    num_iter += 1
    with ti.Tape(loss):
        do_iter()
    if loss[None] > 0.1: update_variables()
    else: break


print("Took ", num_iter, "iterations and ended with loss of ", loss[None])


print("New x is: ", x[None], "New weight is: ", weight[None], "New bias is: ", bias[None], "New result is ", x[None]*weight[None] + bias[None])

