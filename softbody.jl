using LinearAlgebra
using GLMakie
using ForwardDiff
using StaticArrays
using IterativeSolvers

mutable struct Simulation{S <: AbstractFloat,I <: Integer}
    g::S
    dt::S

    X::Vector{S}
    D::Vector{S}
    V::Vector{S}
    M::Vector{S}

    ind::Vector{SVector{3,I}}
    A_inv::Vector{SMatrix{2,2,S,4}}
    vol::Vector{S}
    lambda::Vector{S}
    mu::Vector{S}
end


index_arr(ind) = SA[2 * ind[1] - 1, 2 * ind[1], 2 * ind[2] - 1, 2 * ind[2], 2 * ind[3] - 1, 2 * ind[3]]
edge_mat(x) = SA[x[3] - x[1] x[5] - x[1]; x[4] - x[2] x[6] - x[2]]

function compute_gradient_parallel!(grad, sim::Simulation, D_1, a=0.01)
    N_p = size(sim.X)[1] ÷ 2
    N_t = size(sim.ind)[1]

    lk = ReentrantLock()

    # per-triangle gradient

    Threads.@threads for ti = 1:N_t

        inds = index_arr(sim.ind[ti])

        x_0 = sim.X[inds]
        d_1 = D_1[inds]

        function E(d)
            F = edge_mat(x_0 + d) * sim.A_inv[ti]

            dF = max(det(F), 0.1)

            lJ = dF > a ? log(dF) : log(a) + 1 / a * (dF - a) - 1 / a^2 * (dF - a)^2

            E = sim.mu[ti] * (0.5 * (tr(F' * F) - 3.0) - lJ) + 0.5 * sim.lambda[ti] * lJ^2

            return E
        end

        contrib = sim.dt^2 * sim.vol[ti] * ForwardDiff.gradient(E, d_1)

        lock(lk) do
            grad[inds] += contrib
        end
    end

    # per-vertex gradient

    Threads.@threads for vi = 1:N_p

        inds = SA[2 * vi - 1, 2 * vi]

        x_0 = sim.X[inds]
        v_0 = sim.V[inds]
        d_0 = sim.D[inds]
        d_1 = D_1[inds]
        m = sim.M[vi]

        function E(d)

            I = 0.5 * sum((d - (d_0 + v_0 * sim.dt)).^2) * m

            p = x_0 + d

            E = 0.0

            v = (d .- d_0) / sim.dt

            fh = 0.0

            if p[2] > fh
                E += sim.g * p[2] * m
            else
                E += 100000 * (fh - p[2])^2 + v[1]^2 * (fh - p[2]) * 1000
            end

            return I + sim.dt^2 * E
        end

        contrib = ForwardDiff.gradient(E, d_1)

        lock(lk) do
            grad[inds] += contrib
        end

    end
end

function init_simulation(; g=9.8, dt=0.01, center=[0.5, 0.5], radius=0.5, mu=100, lam=50, n=20)::Simulation

    n_vertices = n + 1
    n_triangles = n

    X = zeros(2 * n_vertices)
    D = zeros(2 * n_vertices)
    V = zeros(2 * n_vertices)
    M = ones(n_vertices)
    
    ind = fill(SA[0,0,0], n_triangles)

    X[[1,2]] = center

    for i = 1:n
        vi = [2 * (i + 1) - 1, 2 * (i + 1)]
        angle = i / n * 2π
        X[vi] .= radius * [cos(angle), sin(angle)] + center
        ind[i] = SA[1, 1 + i, 1 + mod1(i + 1, n)]
    end

    lambda = ones(n_triangles) * lam
    mu = ones(n_triangles) * mu
    A = map(ti -> edge_mat(X[index_arr(ind[ti])]), 1:n_triangles)
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
                    # println("converged after $k iterations in $(i)th pass, |∇E| = $(norm(grad))")
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
    sim = init_simulation(; center=[0.5, 0.5], dt=0.001, lam=20000.0, mu=10000.0, g=10.0, n=4, radius=0.3)

    render_pos = Node(reshape(sim.X + sim.D, 2, size(sim.X)[1] ÷ 2))

    f = Figure(resolution=(500, 500))
    mu_slider = Slider(f[2, 1], range=500:10:10000, startvalue=3000)
    ax = Axis(f[1, 1], aspect=1, limits=(-0.1, 1.1, -0.1, 1.1))

    mesh!(f[1, 1], render_pos, Vector(vcat(sim.ind...)))
    display(f)

    a = time_ns()

    for i = 1:1000000
       
        D = line_search((grad, D_1) -> compute_gradient_parallel!(grad, sim, D_1), [Pass(1.0, 10, sim.D + sim.V * sim.dt, 1e-4), Pass(0.2, 1000, sim.D, 1e-4)])
        sim.V .= (D .- sim.D) ./ sim.dt
        sim.D .= D

        if i % 1000 == 0
            println(1000 / ((time_ns() - a) / 1e9) * sim.dt)
            a = time_ns()
        end

        if i % 30 == 0
            sim.lambda .= mu_slider.value[]
            sim.mu .= mu_slider.value[]
            render_pos[] = reshape(sim.X + sim.D, 2, size(sim.X)[1] ÷ 2)

            sleep(0.01)
        end
    end
end
