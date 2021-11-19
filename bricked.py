import taichi as ti

ti.init(arch=ti.gpu)

x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

def init():
    x[None] = 1.82

@ti.kernel
def calculate_loss(x_val: ti.template()):
    loss[None] = abs(x_val[None])

def update_x():
    x[None] -= 0.001 * x.grad[None]

init()
while x[None] > 0.01:
    with ti.Tape(loss=loss):
        calculate_loss(x)
    update_x()
    print(x[None])

print("x converged")
