using LinearAlgebra
using GLMakie
using ForwardDiff
using StaticArrays
using IterativeSolvers
using SparseArrays

mutable struct Simulation{S <: AbstractFloat,I <: Integer}
    g::S
    floor_force::S
    floor_height::S
    floor_friction::S
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

function triangle_energy(d, A_inv, x_0, a, lambda, mu)
    F = edge_mat(x_0 + d) * A_inv

    dF = det(F)

    lJ = dF > a ? log(dF) : log(a) + 1 / a * (dF - a) - 1 / a^2 * (dF - a)^2

    E = mu * (0.5 * (tr(F' * F) - 3.0) - lJ) + 0.5 * lambda * lJ^2

    return E
end

function vertex_energy(d, d_0, v_0, x_0, dt, m, floor_height, floor_force, floor_friction, g)  
    I = 0.5 * sum((d - (d_0 + v_0 * dt)).^2) * m

    p = x_0 + d

    E = 0.0

    v = (d .- d_0) / dt
    
    if p[2] > floor_height
        E += g * p[2] * m
    else
        E += floor_force * (floor_height - p[2])^2 + v[1]^2 * (floor_height - p[2]) * floor_friction
    end

    return I + dt^2 * E
end

function compute_gradient!(grad, sim::Simulation, D_1, a=0.01)
    N_p = size(sim.X)[1] ÷ 2
    N_t = size(sim.ind)[1]

    lk = ReentrantLock()

    # per-triangle gradient

    Threads.@threads for ti = 1:N_t

        inds = index_arr(sim.ind[ti])

        x_0 = sim.X[inds]
        d_1 = D_1[inds]

        E(d) = triangle_energy(d, sim.A_inv[ti], x_0, a, sim.lambda[ti], sim.mu[ti])

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

        E(d) = vertex_energy(d, d_0, v_0, x_0, sim.dt, sim.M[vi], sim.floor_height, sim.floor_force, sim.floor_friction, sim.g)

        contrib = ForwardDiff.gradient(E, d_1)

    lock(lk) do
            grad[inds] += contrib
    end
    end
end

function init_hessian!(hess, sim::Simulation, init_val=1)

    N_p = size(sim.X)[1] ÷ 2
    N_t = size(sim.ind)[1]
    
    # per-triangle gradient

    for ti = 1:N_t
        inds = index_arr(sim.ind[ti])

        hess[inds, inds] .= init_val
    end

    # per-vertex gradient

    for vi = 1:N_p

        inds = SA[2 * vi - 1, 2 * vi]

        hess[inds, inds] .= init_val
    end

    return hess
end

function compute_hessian!(hess, sim::Simulation, D_1, a=0.01)
    N_p = size(sim.X)[1] ÷ 2
    N_t = size(sim.ind)[1]

    lk = ReentrantLock()

    init_hessian!(hess, sim, 0)

    # per-triangle gradient

    Threads.@threads for ti = 1:N_t

        inds = Vector(index_arr(sim.ind[ti]))

        x_0 = sim.X[inds]
        d_1 = D_1[inds]

        E(d) = triangle_energy(d, sim.A_inv[ti], x_0, a, sim.lambda[ti], sim.mu[ti])

        contrib = sim.dt^2 * sim.vol[ti] * ForwardDiff.hessian(E, d_1)

        lock(lk) do
            hess[inds, inds] .+= contrib
        end
    end

    # per-vertex gradient

    Threads.@threads for vi = 1:N_p

        inds = [2 * vi - 1, 2 * vi]

        x_0 = sim.X[inds]
        v_0 = sim.V[inds]
        d_0 = sim.D[inds]
        d_1 = D_1[inds]

        E(d) = vertex_energy(d, d_0, v_0, x_0, sim.dt, sim.M[vi], sim.floor_height, sim.floor_force, sim.floor_friction, sim.g)

        contrib = ForwardDiff.hessian(E, d_1)

        lock(lk) do
            hess[inds, inds] .+= contrib
        end

    end
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

function newton!(hess, hessian!, gradient!, x_guess, tol)
    
    x = x_guess
    grad = zeros(size(x))

    for i in 1:1000
        grad .= 0
        gradient!(grad, x)
        hessian!(hess, x)

        x .= x .- cg(hess, grad)

        if norm(grad) < tol
            println("Converged after $i iterations.")
            break
        end
    end

    if norm(grad) > tol
println("WARNING: No convergence!")
    end

    return x
end
