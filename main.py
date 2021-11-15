import taichi as ti
import numpy as np
import time


class Mesh:
    def __init__(self):
        self.vertices = []
        self.indices = []
        self.velocities = []
        self.colors = []

    def add_vertex(self,v,p=(0,0), c=(1,1,1)):
        self.vertices.append(v)
        self.velocities.append(p)
        self.colors.append(c)
        return len(self.vertices) - 1
    
    def add_triangle(self,index1, index2, index3):
        self.indices.append(index1)
        self.indices.append(index2)
        self.indices.append(index3)

    def as_np_arrays(self):
        vs = np.empty((len(self.vertices), 2))
        ps = np.empty((len(self.vertices), 2))
        cs = np.empty((len(self.vertices), 3))
        inds = np.empty(len(self.indices))

        for i,(x,y) in enumerate(self.vertices):
            vs[i, 0] = x
            vs[i, 1] = y
            p_x, p_y = self.velocities[i]
            ps[i, 0] = p_x
            ps[i, 1] = p_y
            r,g,b = self.colors[i]
            cs[i, 0] = r
            cs[i, 1] = g
            cs[i, 2] = b




        for i, ind in enumerate(self.indices):
            inds[i] = ind

        return vs, ps, inds, cs

    def plane(center, w, h, size, rot_vel=0):
        mesh = Mesh()

        center_x, center_y = center
        
        s_x = center_x - size * w / 2
        s_y = center_y - size * h / 2

        for i in range(w):
            for j in range(h):
                v_x = s_x + i * size
                v_y = s_y + j * size

                mesh.add_vertex((v_x, v_y), (rot_vel*(v_y-center_y), rot_vel*(center_x-v_x)), (i/w, j/h, 0))

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
        self.colors = ti.Vector.field(3, ti.f32, shape=self.n)

        self.A = ti.Matrix.field(2,2, ti.f32, shape=self.tris)
        self.V = ti.field(ti.f32, shape=self.tris)
        
        self.E = ti.field(ti.f32, shape=(), needs_grad=True)
        self.E_m = ti.field(ti.f32, shape=(), needs_grad=True)
        self.E_c = ti.field(ti.f32, shape=(), needs_grad=True)

        self.indices = ti.field(ti.i32, shape=len(mesh.indices))
        
        vs, ps, inds, cs = mesh.as_np_arrays()

        self.vertices.from_numpy(vs)
        self.indices.from_numpy(inds)
        self.velocities.from_numpy(ps)
        self.colors.from_numpy(cs)

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

       
    @ti.kernel
    def calculate_mesh_energy(self): 
        for tri in range(self.tris):
            x1 = self.vertices[self.indices[3 * tri]] + self.displacement[self.indices[3 * tri]]
            x2 = self.vertices[self.indices[3 * tri + 1]] + self.displacement[self.indices[3 * tri + 1]]
            x3 = self.vertices[self.indices[3 * tri + 2]] + self.displacement[self.indices[3 * tri + 2]]

            e1 = x2-x1
            e2 = x3-x1

            B = ti.Matrix([[e1[0], e2[0]], [e1[1], e2[1]]])
            F = B @ self.A[tri] 
            
            #E = 0.5 * (F.transpose() @ F - ti.Matrix.identity(ti.f32, 2)) #Green Strain not energy
            #self.E[None] += 0.1*((F.determinant() - 1)**2 + (E@E).trace()) * self.V[tri] 

            mu = 0.05
            lam = 0.01

            self.E_m[None] += 0.5 * mu * ((F.transpose() @ F).trace() - 3) - mu * ti.log(F.determinant()) + lam * ti.log(F.determinant())**2 + 0.01 * (x1[1]+x2[1]+x3[1])


    @ti.kernel
    def calculate_energy(self):
        self.E[None] = self.E_m[None]
    
    @ti.kernel
    def step(self):
        for i in self.vertices: 
            self.velocities[i] -= (self.displacement.grad[i] + 0.2 * self.velocities[i]) * self.dt
            self.displacement[i] += self.velocities[i] * self.dt

            if self.positions[i][1] < 0.0:
                self.velocities[i] += ti.Matrix([0.0,1.0]) * self.dt

            self.positions[i] = self.vertices[i] + self.displacement[i] 

if __name__ == "__main__":
    ti.init(arch=ti.cuda) 
    mesh = Mesh.plane((0.5,2.0),3,100,0.02)

    sim = SoftbodySim(mesh, dt = 0.005)

    window = ti.ui.Window("float",res = (1000 , 1000))
    canvas = window.get_canvas()
        
    i = 0
    while window.running:
        sim.E[None] = 0
        sim.E_m[None] = 0
        sim.E_c[None] = 0

        with ti.Tape(sim.E):
            sim.calculate_mesh_energy()
            sim.calculate_energy()
        sim.step()

        canvas.triangles(sim.positions, indices=sim.indices, per_vertex_color=sim.colors)
        window.show()

        i += 1
