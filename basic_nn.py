import taichi as ti
import math
import random

ti.init(arch=ti.gpu)

input_size = 4

x = ti.field(dtype=ti.f32, shape=4, needs_grad=True)
weight = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
bias = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
output = ti.field(dtype=ti.f32, shape=(), needs_grad=True) #Somehow also needs grad even thought its is never accessed

loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


learning_rate = 1e-2


def init():
    for i in range(input_size):
        x[i] = random.random() * 100 - 50
    weight[None] = random.random() * 100 - 50
    bias[None] = random.random() * 100 - 50
    loss[None] = 0.0


@ti.kernel
def calculate_loss():
    loss[None] = abs(output[None] - 5)


def clear_output():
    output[None] = 0


@ti.kernel
def get_output():
    for i in range(input_size):
        output[None] += x[i]*weight[None] + bias[None]


def do_iter():
    get_output()
    #print("Output is ", output[None])
    calculate_loss()
    #print("Loss is", loss[None])


def update_variables():
    for i in range(input_size):
        x[i] -= learning_rate * x.grad[i]
        #print("X_grad is ", x.grad[i])
    weight[None] -= learning_rate * weight.grad[None]
    #print("weight_grad is ", weight.grad[None])
    bias[None] -= learning_rate * bias.grad[None]
    #print("bias_grad is ", bias.grad[None])

init()

#print("Initial x is: ", x[None], "Initial weight is: ", weight[None], "Initial bias is: ", bias[None], "Initial result is", x[None]*weight[None] + bias[None])
print("Starting NN...")
num_iter = 0
while True:
    num_iter += 1
    if num_iter % 100 == 0: print("Loss at iteration ", num_iter, " is ", loss[None])
    clear_output()
    with ti.Tape(loss):
        do_iter()
    if loss[None] > 0.1: update_variables()
    else: break


print("Took ", num_iter, "iterations and ended with loss of ", loss[None])


#print("New x is: ", x[None], "New weight is: ", weight[None], "New bias is: ", bias[None], "New result is ", x[None]*weight[None] + bias[None])
