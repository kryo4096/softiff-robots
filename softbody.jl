using LinearAlgebra
using GLMakie
using ForwardDiff

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

function compute_gradient_parallel!(grad, sim::Simulation, D_1, a = 0.01)
    N_p = size(sim.X)[2]
    N_t = size(sim.ind)[2]

    lk = ReentrantLock()

    # per-triangle gradient

    Threads.@threads for ti = 1:N_t
        x_0 = sim.X[:, sim.ind[:, ti]]
        d_1 = D_1[:, sim.ind[:, ti]]

        function E(d)
            F = edge_mat(x_0 + d) * sim.A_inv[ti]

            dF = max(det(F), 0.1)

            lJ = dF > a ? log(dF) : log(a) + 1 / a * (dF - a) - 1 / a^2 * (dF - a)^2

            E = sim.mu[ti] * (0.5 * (tr(F' * F) - 3.0) - lJ) + 0.5 * sim.lambda[ti] * lJ^2

            return E
        end

        contrib = sim.dt^2 * sim.vol[ti] * ForwardDiff.gradient(E, d_1)

        lock(lk) do
            grad[:, sim.ind[:, ti]] += contrib
        end
    end

    # per-vertex gradient

    Threads.@threads for vi = 1:N_p
        x_0 = sim.X[:, vi]
        v_0 = sim.V[:, vi]
        d_0 = sim.D[:, vi]
        d_1 = D_1[:, vi]
        m = sim.M[vi]

        function E(d)

            I = 0.5 * sum((d - (d_0 + v_0 * sim.dt)) .^ 2) * m

            p = x_0 + d

            if p[2] > 0.0
                return I + sim.dt^2 * sim.g * p[2] * m
            else
                return I + sim.dt^2 * 10000 * (0.0 - p[2])^2
            end
        end

        contrib = ForwardDiff.gradient(E, d_1)

        lock(lk) do
            grad[:, vi] += contrib
        end

    end
end

function init_simulation(; g = 9.8, dt = 0.01, center = [0.5, 0.5], radius = 0.5, mu = 100, lam = 50, n = 20)::Simulation

    n_vertices = n + 1
    n_triangles = n

    X = zeros(2, n_vertices)
    D = zeros(2, n_vertices)
    V = zeros(2, n_vertices)
    M = ones(n_vertices)

    V[2, :] .-= 5

    ind = zeros(Int, 3, n_triangles)

    X[:, 1] = center

    for i = 1:n
        angle = i / n * 2π
        X[:, i+1] = radius * [cos(angle), sin(angle)] + center
        # V[:, i + 1] = 10 * [-sin(angle), cos(angle)]
        ind[:, i] = [1, 1 + i, 1 + mod1(i + 1, n)]
    end

    lambda = ones(n_triangles) * lam
    mu = ones(n_triangles) * mu
    A = map(ti -> edge_mat(X[:, ind[:, ti]]), 1:n_triangles)
    A_inv = inv.(A)

    vol = abs.(0.5 * det.(A))

    return Simulation(g, dt, X, D, V, M, ind, A_inv, vol, lambda, mu)
end

function main()
    sim = init_simulation(; center = [0.5, 0.5], dt = 0.001, lam = 20000, mu = 10000, g = 10, n = 50, radius = 0.3)

    render_pos = Node(sim.X + sim.D)

    f = Figure(resolution = (500, 500))
    mu_slider = Slider(f[2, 1], range = 10:10:30000, startvalue = 3000)
    ax = Axis(f[1, 1], aspect = 1, limits = (-0.1, 1.1, -0.1, 1.1))

    mesh!(f[1, 1], render_pos, sim.ind[:])
    display(f)

    avg_time = 1 / 30.0

    last_frame = 0

    grad = zeros(size(sim.D))
    D_1 = zeros(size(sim.D))

    compute_gradient_parallel!(grad, sim, D_1)
    fill!(grad, 0.0)
    @time compute_gradient_parallel!(grad, sim, D_1)



    for i = 1:1000000
        D_1[:] = sim.D + sim.V * sim.dt

        converged = false

        for k = 1:10
            fill!(grad, 0.0)
            compute_gradient_parallel!(grad, sim, D_1)

            D_1[:] = D_1 - 1.0 * grad

            if norm(grad) < 1e-3
                converged = true
                break
            end
        end

        if !converged
            D_1[:] = sim.D

            for k = 1:100
                fill!(grad, 0.0)
                compute_gradient_parallel!(grad, sim, D_1)

                D_1 = D_1 - 0.1 * grad

                if norm(grad) < 1e-3
                    converged = true
                    break
                end
            end
        end

        if !converged
            println("WARNING! no convergence")
        end

        sim.V[:] = ((D_1-sim.D)/sim.dt)[:]
        sim.D[:] = D_1[:]

        if i % 10 == 0

            sim.lambda .= mu_slider.value[]
            sim.mu .= mu_slider.value[]
            render_pos[] = sim.X + sim.D

            sleep(0.0001)
        end
    end
end

