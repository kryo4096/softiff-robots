import taichi as ti
import numpy as np
import math
import random


import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)

input_size = 4
hidden_layer_size = 10
output_size = 4

input = ti.field(dtype=ti.f32, shape=input_size, needs_grad=True)
weight_input = ti.Vector.field(input_size, dtype=ti.f32, shape=hidden_layer_size, needs_grad=True)
bias_input = ti.Vector.field(input_size, dtype=ti.f32, shape=hidden_layer_size, needs_grad=True)
hidden_layer = ti.field(dtype=ti.f32, shape=hidden_layer_size, needs_grad=True)
weight_hidden_layer = ti.Vector.field(hidden_layer_size, dtype=ti.f32, shape=output_size, needs_grad=True)
bias_hidden_layer = ti.Vector.field(hidden_layer_size, dtype=ti.f32, shape=output_size, needs_grad=True)
output = ti.field(dtype=ti.f32, shape=output_size, needs_grad=True) #Somehow also needs grad even thought its is never accessed

loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


#learning_rate = 5e-4

dt = 3e-3
g = -9.81


def init():
    for i in range(input_size):
        input[i] = random.random() #* 10 - 5
        for j in range(hidden_layer_size):
            weight_input[j][i] = random.random() #* 10 - 5
            bias_input[j][i] = random.random() #* 10 - 5

    for i in range(output_size):
        for j in range(hidden_layer_size):
            weight_hidden_layer[j][i] = random.random() #* 10 - 5
            bias_hidden_layer[j][i] = random.random() #* 10 - 5
        output[i] = 0

    loss[None] = 0.0


@ti.func
def check_initial_conditions():
    x, y, v_x, v_y = output[0], output[1], output[2], output[3]
    return_value = 0.0
    if x < 0:
        #print("Violated lower x")
        return_value += x
    elif x > 10:
        #print("Violated upper x")
        return_value += 10-x # x optimum at 100
    if y < 0:
        #print("Violated lower y")
        return_value += y
    elif y > 10:
        #print("Violated upper y")
        return_value += 10-y # y optimum at 100
    if v_x < 0:
        #print("Violated lower v_x")
        return_value += v_x # velocity should be in the positive x-direction
    if v_x**2> 10:
        #print("Violated upper vel x")
        return_value += 10-v_x**2 ## velocity should not be too high
    if v_y**2 > 10:
        return_value += 10-v_y**2

    #Guarantee Smoothness for function if initial conditions are violated!

    '''
    if return_value == 0: print("No initial conditions violated for simulation")
    else: print("Initial condition violation of magnitude ", return_value)
    '''
    return return_value


@ti.func
def simulate():
    x, y, v_x, v_y = output[0], output[1], output[2], output[3]
    return_value = 0.0
    initial_cond = check_initial_conditions()
    #print("Initial iteration values: ", x, y, v_x, v_y)
    for _ in ti.static(range(300)):
        v_y += g*dt
        x += v_x*dt
        y += v_y*dt
        if return_value == 0 and y < 0:
            #print("Simulation reached distance of ", x)
            return_value = x
    if return_value == 0 and y > 0:
        return_value = max(x-y, 0)

    #print("Final iteration values:   ", x, y, v_x, v_y, "and ", return_value)
    return return_value + initial_cond*10

def graphical_simulate(x, y, v_x, v_y):
    return_value = 0.0
    #initial_cond = check_initial_conditions()
    print("Initial iteration values: ", x, y, v_x, v_y)
    gui = ti.GUI("NeuralNetwork Simulator")
    for _ in range(1000):
        v_y += g*dt
        x += v_x*dt
        y += v_y*dt
        if return_value == 0 and y < 0:
            #print("Simulation reached distance of ", x)
            return_value = x
        l_start = np.array([0, 0.5])
        p = np.array([x/40+0.5, y/40+0.5])
        gui.circle(p, radius=10, color=0xFF0000)

        gui.show()

    if return_value == 0 and y > 0:
        return_value = max(x-y, 0)

    print("Final iteration values:   ", x, y, v_x, v_y, "and ", return_value)


@ti.kernel
def calculate_loss():
    value = simulate()
    loss[None] = -value


@ti.kernel
def clear_variables():
    loss[None] = 0
    for i in ti.static(range(hidden_layer_size)):
        hidden_layer[i] = 0
    for i in ti.static(range(output_size)):
        output[i] = 0


@ti.kernel
def get_output():
    for i in ti.static(range(input_size)):
        for j in ti.static(range(hidden_layer_size)):
            hidden_layer[j] += input[i] * weight_input[j][i] + bias_input[j][i]
    for i in ti.static(range(output_size)):
        for j in ti.static(range(hidden_layer_size)):
            output[i] += hidden_layer[j] * weight_hidden_layer[j][i] + bias_hidden_layer[j][i]


def do_iter():
    get_output()
    #print("Output is ", output[None])
    calculate_loss()
    #print("Loss is", loss[None])


def update_variables(index):
    learning_rate = 1e-5
    for i in range(input_size):
        for j in range(hidden_layer_size):
            weight_input[j][i] -= learning_rate * weight_input.grad[j][i]
            bias_input[j][i] -= learning_rate * bias_input.grad[j][i]
    for i in range(hidden_layer_size):
        for j in range(output_size):
            weight_hidden_layer[i][j] -= learning_rate * weight_hidden_layer.grad[i][j]
            bias_hidden_layer[i][j] -= learning_rate * bias_hidden_layer.grad[i][j]


init()

graphical_simulate(9.93513298034668, 5.346199035644531, 2.7409629821777344, -1.5206743478775024)

#print("Initial x is: ", x[None], "Initial weight is: ", weight[None], "Initial bias is: ", bias[None], "Initial result is", x[None]*weight[None] + bias[None])
print("Starting NN...")
loss_arr = []
iterations = []
for i in range(1000):
    loss_arr.append(loss[None])
    iterations.append(i)

    clear_variables()
    with ti.Tape(loss):
        do_iter()

    if i % 100 == 0:
        print("Loss at iteration ", i, " is ", loss[None])
        print("Final values are ", output[0], output[1], output[2], output[3])

    update_variables(i)


print("Ended with loss of ", loss[None])
print("Ended with parameters", output[0], output[1], output[2], output[3], math.atan2(output[3], output[2]))

graphical_simulate(output[0], output[1], output[2], output[3])

plt.plot(iterations, loss_arr)
plt.show()

#print("New x is: ", x[None], "New weight is: ", weight[None], "New bias is: ", bias[None], "New result is ", x[None]*weight[None] + bias[None])
