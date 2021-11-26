using LinearAlgebra
using GLMakie
using ForwardDiff
using StaticArrays
using IterativeSolvers

mutable struct Simulation{S <: AbstractFloat,I <: Integer}
    g::S
    dt::S

    X::Matrix{S}
    D::Matrix{S}
    V::Matrix{S}
    M::Vector{S}

    ind::Matrix{I}
    A_inv::Vector{SMatrix{2,2,S,4}}
    vol::Vector{S}
    lambda::Vector{S}
    mu::Vector{S}
end


edge_mat(x) = SMatrix{2,2}([x[:, 2] - x[:, 1] x[:, 3] - x[:, 1]])

function compute_gradient_parallel!(grad, sim::Simulation, D_1, a=0.01)
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

            I = 0.5 * sum((d - (d_0 + v_0 * sim.dt)).^2) * m

            p = x_0 + d

            E = 0.0

            v = (d .- d_0) / sim.dt

            fh = 0.1

            if p[2] > fh
                E += sim.g * p[2] * m
            else
                E += 100000 * (fh- p[2])^2 + v[1]^2 * (fh - p[2]) * 100
            end

            return I + sim.dt^2 * E
        end

        contrib = ForwardDiff.gradient(E, d_1)

        lock(lk) do
            grad[:, vi] += contrib
        end

    end
end
    
function compute_hxv!(V, sim::Simulation, D_1, a=0.01)
    N_p = size(sim.X)[2]
    N_t = size(sim.ind)[2]

    lk = ReentrantLock()

    prod = zeros(size(sim.X))

    # per-triangle gradient

    Threads.@threads for ti = 1:N_t
        x_0 = sim.X[:, sim.ind[:, ti]]
        d_1 = D_1[:, sim.ind[:, ti]]
        v = V[:, sim.ind[:, ti]]

        function E(d)
            F = edge_mat(x_0 + d) * sim.A_inv[ti]

            dF = max(det(F), 0.1)

            lJ = dF > a ? log(dF) : log(a) + 1 / a * (dF - a) - 1 / a^2 * (dF - a)^2

            E = sim.mu[ti] * (0.5 * (tr(F' * F) - 3.0) - lJ) + 0.5 * sim.lambda[ti] * lJ^2

            return sim.dt^2 * sim.vol[ti] * E
        end

        contrib = gradient(d -> sum(ForwardDiff.gradient(E, d) .* v, d_1))

        lock(lk) do
            prod[:, sim.ind[:, ti]] += contrib
        end
    end

    # per-vertex gradient

    Threads.@threads for vi = 1:N_p
        x_0 = sim.X[:, vi]
        v_0 = sim.V[:, vi]
        d_0 = sim.D[:, vi]
        d_1 = D_1[:, vi]
        m = sim.M[vi]
        v = V[:, sim.ind[:, ti]]

        function E(d)

            I = 0.5 * sum((d - (d_0 + v_0 * sim.dt)).^2) * m

            p = x_0 + d

            E = 0.0

            v .= (d .- d_0) / sim.dt

            if p[2] > 0.0
                E += sim.g * p[2] * m
            else
                E += 100000 * (0.0 - p[2])^2 - v[1]^2 * p[2] * 1000
            end

            return I + sim.dt^2 * E
        end

         contrib = gradient(d -> sum(ForwardDiff.gradient(E, d) .* v, d_1))

        lock(lk) do
            prod[:, vi] += contrib
        end

    end

    return prod
end

function init_simulation(; g=9.8, dt=0.01, center=[0.5, 0.5], radius=0.5, mu=100, lam=50, n=20)::Simulation

    n_vertices = n + 1
    n_triangles = n

    X = zeros(2, n_vertices)
    D = zeros(2, n_vertices)
    V = zeros(2, n_vertices)
    M = ones(n_vertices)

    ind = zeros(Int, 3, n_triangles)

    X[:, 1] = center

    for i = 1:n
        angle = i / n * 2π
        X[:, i + 1] = radius * [cos(angle), sin(angle)] + center
        V[:, i + 1] = 5 * [-sin(angle), cos(angle)] + [0,1]
        ind[:, i] = [1, 1 + i, 1 + mod1(i + 1, n)]
    end

    lambda = ones(n_triangles) * lam
    mu = ones(n_triangles) * mu
    A = map(ti -> edge_mat(X[:, ind[:, ti]]), 1:n_triangles)
    A_inv = inv.(A)

    vol = abs.(0.5 * det.(A))

    return Simulation(g, dt, X, D, V, M, ind, A_inv, vol, lambda, mu)
end

struct Pass
    alpha
    iter
    guess
    tol
end

function line_search(gradient, passes::Vector{Pass})
        converged = false

        grad = zeros(size(passes[1].guess))
        x = zeros(size(passes[1].guess))

        for (i, pass) in enumerate(passes)
            if converged 
                break
            end

            x .= pass.guess

            for k = 1:pass.iter
                grad .= 0.0
                gradient(grad, x)

                x .= x .- grad .* pass.alpha

                if norm(grad) < pass.tol
                    println("converged after $k iterations in $(i)th pass, |∇E| = $(norm(grad))")
                    converged = true
                    break
                end
            end

        end

        if !converged
            println("WARNING! no convergence")
        end

        return x
end

function main()
    sim = init_simulation(; center=[0.5, 0.5], dt=0.001, lam=20000.0, mu=10000.0, g=10.0, n=10, radius=0.3)

    render_pos = Node(sim.X + sim.D)

    f = Figure(resolution=(500, 500))
    mu_slider = Slider(f[2, 1], range=500:10:10000, startvalue=3000)
    ax = Axis(f[1, 1], aspect=1, limits=(-0.1, 1.1, -0.1, 1.1))

    mesh!(f[1, 1], render_pos, sim.ind[:])
    display(f)

    for i = 1:1000000
       
        D = line_search((grad, D_1) -> compute_gradient_parallel!(grad, sim, D_1), [Pass(1.0, 10, sim.D + sim.V * sim.dt, 1e-3), Pass(0.2, 100, sim.D, 1e-3)])
        sim.V .= (D .- sim.D) ./ sim.dt
        sim.D .= D

        if i % 1 == 0
            sim.lambda .= mu_slider.value[]
            sim.mu .= mu_slider.value[]
            render_pos[] = sim.X + sim.D
            sleep(0.001)
        end
    end
end

main()