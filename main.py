import time
import math
import colorsys

import taichi as ti
import numpy as np


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

    def disk(center, n, r):
        mesh = Mesh()

        mesh.add_vertex(center)

        for i in range(n):
            angle = 2 * math.pi * i / n
                
            col = colorsys.hls_to_rgb(i/n, 0.5, 1)

            mesh.add_vertex((center[0] + math.cos(angle) * r, center[1] + math.sin(angle) * r), c=col) 
            mesh.add_triangle(0, i+1, 1 + (i+1)%n)

        return mesh

            

@ti.data_oriented
class SoftbodySim:
    def __init__(self,mesh, dt=0.1, damping=0.1, g = 0.098):
        self.dt = dt # time step
        self.damping = damping #damping coefficient
        self.g = g # gravity
        
        # constant fields (given by user)
        self.n = len(mesh.vertices) #number of vertices in the softbody
        self.tris = int(len(mesh.indices) / 3) #number of triangles
        self.vertices = ti.Vector.field(2, ti.f32, shape=self.n) # undeformed positions
        self.indices = ti.field(ti.i32, shape=len(mesh.indices)) # triangle index list
        self.colors = ti.Vector.field(3, ti.f32, shape=self.n) # vertex colors
       
        # initialize constant fields from mesh object
        vs, ps, inds, cs = mesh.as_np_arrays()

        self.vertices.from_numpy(vs)
        self.indices.from_numpy(inds)
        self.colors.from_numpy(cs)

        # precomputable fields
        self.V = ti.field(ti.f32, shape=self.tris) # triangle areas
        self.A = ti.Matrix.field(2,2, ti.f32, shape=self.tris) #triangle mappings (world vectors -> reference triangle vectors)

        self.precompute()

        # dynamic fields
        self.displacement = ti.Vector.field(2, ti.f32, shape=self.n, needs_grad=True) # displacement from original positions
        self.velocities = ti.Vector.field(2, ti.f32, shape=self.n) # time derivative of displacement
        self.positions = ti.Vector.field(2, ti.f32, shape=self.n) # original positions + displacement (real position)
        self.E = ti.field(ti.f32, shape=(), needs_grad=True) # total energy of the system, is autodifferentiated to compute forces
        self.t =ti.field(ti.f32, shape=())
        
        # user can supply initial velocities
        self.velocities.from_numpy(ps)

   
    @ti.kernel
    def precompute(self):
        for tri in range(self.tris):
            # obtain triangle corners
            x1 = self.vertices[self.indices[3 * tri]]
            x2 = self.vertices[self.indices[3 * tri + 1]]
            x3 = self.vertices[self.indices[3 * tri + 2]]
            
            # compute any two edge vectors
            e1 = x2-x1
            e2 = x3-x1
            
            # compute mapping (reference triangle vector -> world vector)
            A = ti.Matrix([[e1[0], e2[0]], [e1[1], e2[1]]])

            # invert it
            self.A[tri] = A.inverse()

            # compute area from cross product of edges
            self.V[tri] = ti.abs(A.determinant()) / 2

       
    @ti.kernel
    def calculate_energy(self): 
        for tri in range(self.tris):

            # retrieve the corners of the current element / triangle
            x1 = self.vertices[self.indices[3 * tri]] + self.displacement[self.indices[3 * tri]]
            x2 = self.vertices[self.indices[3 * tri + 1]] + self.displacement[self.indices[3 * tri + 1]]
            x3 = self.vertices[self.indices[3 * tri + 2]] + self.displacement[self.indices[3 * tri + 2]]
            
            #calculate the same two edges of the triangle as for the matrix A above
            e1 = x2-x1
            e2 = x3-x1
            
            # Matrix B mapping reference triangle to deformed triangle
            B = ti.Matrix([[e1[0], e2[0]], [e1[1], e2[1]]])

            # The deformation gradient matrix F is defined as the matrix product of B and A^-1, 
            # with A^-1 being the matrix mapping the current undeformed triangle to the reference triangle
            # A can be precomputed
            F = B @ self.A[tri] 
            
            #Lamé coefficients
            mu = 100 # shear viscosity mu
            lam = 100 # 1st lamé parameter lambda (no physical intuition)

            # Calculate energy contribution from neo-hookean energy density
            self.E[None] += (0.5 * mu * ((F.transpose() @ F).trace() - 3) - mu * ti.log(F.determinant()) + lam * ti.log(F.determinant())**2) * self.V[tri]
            
            midpoint = (x1 + x2 + x3) / 3


            # and from gravity (floor forces are accounted for later)
            self.E[None] += self.g * midpoint[1] * self.V[tri] 


    @ti.kernel
    def step(self):
        for i in self.vertices:
            # update velocities according to potential gradient and damp it slightly
            self.velocities[i] -= (self.displacement.grad[i] + self.damping * self.velocities[i]) * self.dt

            # update displacements according to velocities
            self.displacement[i] += self.velocities[i] * self.dt
            
            # floor force
            if self.positions[i][1] < 0.0:
                self.velocities[i] += ti.Matrix([0.0,1.0]) * self.dt
            
            # compute real position for rendering
            self.positions[i] = self.vertices[i] + self.displacement[i] 

            self.t[None] += self.dt

if __name__ == "__main__":
    ti.init(arch=ti.cuda) 
    #mesh = Mesh.plane((0.5,0.5),6,20,0.01)

    mesh = Mesh.disk((0.5,1.0), 200, 0.4)
    sim = SoftbodySim(mesh, dt = 0.001, damping=0.5, g=90)

    window = ti.ui.Window("float",res = (1000 , 1000))
    canvas = window.get_canvas()
        
    i = 0
    while window.running:
        with ti.Tape(sim.E):
            sim.calculate_energy()
        sim.step()

        canvas.triangles(sim.positions, indices=sim.indices, per_vertex_color=sim.colors)
        window.show()

        i += 1
