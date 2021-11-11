import taichi as ti
import numpy as np
import time


class Mesh:
    def __init__(self):
        self.vertices = []
        self.indices = []

    def add_vertex(self,v):
        self.vertices.append(v)
        return len(self.vertices) - 1
    
    def add_triangle(self,index1, index2, index3):
        self.indices.append(index1)
        self.indices.append(index2)
        self.indices.append(index3)

    def as_np_arrays(self):
        vs = np.empty((len(self.vertices), 2))
        inds = np.empty(len(self.indices))

        for i,(x,y) in enumerate(self.vertices):
            vs[i, 0] = x
            vs[i, 1] = y

        for i, ind in enumerate(self.indices):
            inds[i] = ind

        return vs, inds

    def plane(offset, w, h, size):
        mesh = Mesh()

        for i in range(w):
            for j in range(h):
                mesh.add_vertex((offset + i * size, offset + j * size));
                
                if i < w - 1 and j < h - 1:
                    mesh.add_triangle((i + 1) * h + j, i * h + j, i * h + j + 1)
                    mesh.add_triangle((i + 1) * h + j, i * h + j + 1, (i+1)*h + j + 1)

        return mesh

@ti.data_oriented
class SoftbodySim:
    def __init__(self,mesh, dt=0.1):
        self.dt = dt
        self.n = len(mesh.vertices)
        self.tris = int(len(mesh.indices) / 3)
        self.vertices = ti.Vector.field(2, ti.f32, shape=self.n)
        self.displacement = ti.Vector.field(2, ti.f32, shape=self.n, needs_grad=True)
        self.velocities = ti.Vector.field(2, ti.f32, shape=self.n)
        self.positions = ti.Vector.field(2, ti.f32, shape=self.n)

        self.A = ti.Matrix.field(2,2, ti.f32, shape=self.tris)
        self.V = ti.field(ti.f32, shape=self.tris)
        self.E = ti.field(ti.f32, shape=(), needs_grad=True)

        self.indices = ti.field(ti.i32, shape=len(mesh.indices))
        
        vs, inds = mesh.as_np_arrays()

        self.vertices.from_numpy(vs)
        self.indices.from_numpy(inds)

        self.init()
   
    @ti.kernel
    def init(self):
        for tri in range(self.tris):
            x1 = self.vertices[self.indices[3 * tri]]
            x2 = self.vertices[self.indices[3 * tri + 1]]
            x3 = self.vertices[self.indices[3 * tri + 2]]

            e1 = x2-x1
            e2 = x3-x1
            A = ti.Matrix([[e1[0], e2[0]], [e1[1], e2[1]]])
            self.A[tri] = A.inverse()
            self.V[tri] = (e1[0] * e2[1] - e1[1] * e2[0]) / 2
            print(A)
            print(self.A[tri])
            print(self.V[tri])

       
    @ti.kernel
    def calculate_energy(self): 
        for tri in range(self.tris):
            x1 = self.vertices[self.indices[3 * tri]] + self.displacement[self.indices[3 * tri]]
            x2 = self.vertices[self.indices[3 * tri + 1]] + self.displacement[self.indices[3 * tri + 1]]
            x3 = self.vertices[self.indices[3 * tri + 2]] + self.displacement[self.indices[3 * tri + 2]]

            e1 = x2-x1
            e2 = x3-x1

            B = ti.Matrix([[e1[0], e2[0]], [e1[1], e2[1]]])
            F = B @ self.A[tri] 
            
            E = 0.5 * (F.transpose() @ F - ti.Matrix.identity(ti.f32, 2)) #Green Strain not energy
            self.E[None] += (0.05*(E.trace())**2 + 0.1*(E@E).trace()) * self.V[tri] 

    @ti.kernel
    def step(self):
        for i in self.vertices: 
            self.velocities[i] -= (self.displacement.grad[i] + 10 * self.velocities[i]) * self.dt
            self.displacement[i] += self.velocities[i] * self.dt

            if self.positions[i][1] < 0.1:
                self.velocities[i] += ti.Matrix([0.0,0.1]) * self.dt
            else:
                self.velocities[i] -= ti.Matrix([0.0,0.01]) * self.dt

            self.positions[i] = self.vertices[i] + self.displacement[i] 

if __name__ == "__main__":
    ti.init() 
    mesh = Mesh.plane(0.3,10,10,0.04)

    sim = SoftbodySim(mesh, dt=0.01)

    window = ti.ui.Window("float",res = (500,500))
    canvas = window.get_canvas()
        
    i = 0
    while window.running:
        sim.E[None] = 0
        with ti.Tape(sim.E):
            sim.calculate_energy()
        sim.step()

        canvas.triangles(sim.positions, color=(1.0,1.0,1.0), indices=sim.indices)
        window.show()

        i += 1
