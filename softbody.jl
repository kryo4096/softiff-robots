using LinearAlgebra
using Zygote
using GLMakie
using NLsolve

mutable struct Simulation
    g
    dt
   
    X
    D
    V
    M
      
    ind
    A_inv
    vol
    lambda
    mu
end


edge_mat(x) = [x[:, 2] - x[:, 1] x[:, 3] - x[:, 1]]

function compute_gradient(sim::Simulation, D_1)
    N_p = size(sim.X)[2]
    N_t = size(sim.ind)[2]

    energy_grad = zeros(2, N_p)
    inertia_grad = zeros(2, N_p)

    # per-triangle gradient

    for ti in 1:N_t
        x_0 = sim.X[:, sim.ind[:,ti]]
        d_1 = D_1[:, sim.ind[:,ti]]

        function E(d) 
            F = edge_mat(x_0 + d) * sim.A_inv[ti]
   
            lJ = log(abs(det(F)))

            E =  sim.mu[ti] * (0.5 * (tr(F' * F) - 3.) - lJ) + 0.5 * sim.lambda[ti] * lJ^2

            return E
        end

        energy_grad[:, sim.ind[:, ti]] += sim.vol[ti] * gradient(E, d_1)[1]
    end

    for vi in 1:N_p
        x_0 = sim.X[:, vi]
        v_0 = sim.V[:, vi]
        d_0 = sim.D[:, vi]
        d_1 = D_1[:, vi]
        m = sim.M[vi]

        I(d) = 0.5 * sum((d - (d_0 + v_0 * sim.dt)).^2) * m
            
        inertia_grad[:, vi] += gradient(I, d_1)[1]

        function E(d) 
            p = x_0 + d
            
            if p[2] > 0.1
                return sim.g * p[2] * m
            else
                return 10000 * (0.1 - p[2])^2
            end 
        end

        energy_grad[:, vi] += gradient(E, d_1)[1]

    end

    return inertia_grad + sim.dt^2 * energy_grad
end

function init_simulation(;g=9.8, dt=0.01, center=[0.5,0.5], radius=0.5, mu=100, lam=50, n=20)::Simulation
    
    n_vertices = n + 1
    n_triangles = n

    X = zeros(2, n_vertices)
    D = zeros(2, n_vertices)
    V = zeros(2, n_vertices)
    M = ones(n_vertices)

    V[2,:] .-= 5

    ind = zeros(Int, 3, n_triangles)

    X[:, 1] = center
    
    for i in 1:n
        angle = i / n * 2π
        X[:, i + 1] = radius * [cos(angle), sin(angle)] + center
        V[:, i + 1] = 10 * [-sin(angle), cos(angle)]
        ind[:,i] = [1, 1 + i, 1 + mod1(i + 1, n)]
    end

    lambda = ones(n_triangles) * lam
    mu = ones(n_triangles) * mu
    A = map(ti -> edge_mat(X[:, ind[:,ti]]), 1:n_triangles)
    A_inv = inv.(A)
    
    vol = abs.(0.5 * det.(A))

    return Simulation(g, dt, X, D, V, M, ind, A_inv, vol, lambda, mu)
end

function main()
    sim = init_simulation(;center=[0.5,0.5], dt=0.003, lam=20000, mu=10000, g=10, n=10, radius=0.3)
    sim.D = zeros(size(sim.D))

    render_pos = Node(sim.X + sim.D)

    f = Figure(resolution=(500, 500))
    ax = Axis(f[1, 1], aspect=1, limits=(0., 1., 0., 1.))

    mesh!(f[1,1], render_pos, sim.ind[:])
    display(f) 

    z = 3

    for i in 1:10000

        d_1_far = sim.D + 1.0 * sim.V * sim.dt
        d_1_near = sim.D

        d_1 = d_1_near

        converged = false

        for k in 1:100

            if k < 5
                grad_far = compute_gradient(sim, d_1_far)
                d_1_far -= grad_far

            
                if norm(grad_far) < 1e-3
                    d_1 = d_1_far
                    converged = true

                    println("Far guess converged after $k iterations")
                    break 
                end
            end
            

            if k > 2
                grad_near = compute_gradient(sim, d_1_near)

                d_1_near -= 0.1 * grad_near

                if norm(grad_near) < 1e-3
                    d_1 = d_1_near
                    converged = true

                    println("Near guess converged after $k iterations")
                    break 
                end   
            end  
        end

        it = 1
        alpha = 1.0
        converged = false

        if !converged
            println("WARNING! no convergence")
        end

        sim.V = (d_1 - sim.D) / sim.dt
        sim.D = d_1
        
        render_pos[] = sim.X + sim.D

        sleep(0.01)
    end
end

