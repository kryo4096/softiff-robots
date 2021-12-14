module Softbody
    using LinearAlgebra
    using GLMakie
    using ForwardDiff
    using StaticArrays
    using IterativeSolvers
    using SparseArrays
    using Printf

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

        actuators_i::Vector{I}
        actuators_j::Vector{I}
        spring_stiffness::S
    end 

    function create_simulation(vertices::Vector, indices::Vector, actuators::Vector, ;g=20.0, floor_force=5e3, floor_height=0.0, floor_friction=5e3, dt=0.005, lambda=5e3, mu=5e3, spring_stiffness=1e3, m = 1)
        D = zeros(size(vertices))
        V = zeros(size(vertices))
        M = ones(size(vertices)[1]÷2) * m

        ind = [SA[indices[i], indices[i+1], indices[i+2]] for i in 1:3:length(indices)]
        actuators_i = [i for (i,_) in actuators]
        actuators_j = [j for (_,j) in actuators]

        lambda_vec = ones(size(ind)) * lambda
        mu_vec = ones(size(ind)) * mu

        A = map(ti -> edge_mat(vertices[index_arr(ind[ti])]), 1:length(ind))
        A_inv = Vector(inv.(A))
        vol = abs.(0.5 * det.(A))

        Simulation(g, floor_force, floor_height, floor_friction, dt, vertices, D, V, M, ind, A_inv, vol, lambda_vec, mu_vec, actuators_i, actuators_j, spring_stiffness)
    end

    function render_verts(sim::Simulation)
        reshape(sim.X + sim.D, 2, length(sim.X)÷2)
    end

    function relu(x)
        if x < 0
            return 0
        else
            return x 
        end
    end
    

    index_arr(ind) = SA[2 * ind[1] - 1, 2 * ind[1], 2 * ind[2] - 1, 2 * ind[2], 2 * ind[3] - 1, 2 * ind[3]]
    edge_mat(x) = SA[x[3] - x[1] x[5] - x[1]
                    x[4] - x[2] x[6] - x[2]]

    function triangle_energy(d, A_inv, x_0, a, lambda, mu)
        F = edge_mat(x_0 + d) * A_inv
        dF = det(F)
        lJ = dF > a ? log(dF) : log(a) + 1 / a * (dF - a) - 1 / a^2 * (dF - a)^2
        E = mu * (0.5 * (tr(F' * F) - 3.0) - lJ) + 0.5 * lambda * lJ^2
        return E
    end

    function actuation_energy(d, spring_stiffness, x_0, actuation, dt)
        initial_length = sqrt((x_0[3]-x_0[1])^2 + (x_0[4]-x_0[2])^2)

        d_p = x_0 + d

        current_length = sqrt((d_p[3]-d_p[1])^2 + (d_p[4]-d_p[2])^2)

        E = 0.5 * spring_stiffness * (current_length/(initial_length * actuation) - 1)^2

        return E
    end

    function vertex_energy(d, d_0, v_0, x_0, dt, m, floor_height, floor_force, floor_friction, g)  
        I = 0.5 * sum((d - (d_0 + v_0 * dt)).^2) * m
        p = x_0 + d
        E = 0.0
        v = (d .- d_0) / dt

        E += g * m * relu(p[2] - floor_height)
        E += floor_force * relu(floor_height - p[2])^2 + v[1]^2 * relu(floor_height - p[2]) * floor_friction

        return I / dt^2 + E
    end

    function compute_gradient!(grad, sim::Simulation, D_1, A, a=0.01)
        N_p = size(sim.X)[1] ÷ 2
        N_t = size(sim.ind)[1]
        N_a = size(A)[1]

        # per-triangle gradient

        for ti = 1:N_t

            inds = index_arr(sim.ind[ti])

            x_0 = sim.X[inds]
            d_1 = D_1[inds]

            E(d) = triangle_energy(d, sim.A_inv[ti], x_0, a, sim.lambda[ti], sim.mu[ti])

            grad[inds] += sim.vol[ti] * ForwardDiff.gradient(E, d_1)
        end
        
        # per-actuation gradient
        for ai = 1:N_a
            row = sim.actuators_i[ai]
            col = sim.actuators_j[ai]
            act = A[ai]

            inds = SA[2 * col - 1, 2 * col, 2*row - 1, 2*row]

            x_0 = sim.X[inds]
            d_1 = D_1[inds]

            E(d) = actuation_energy(d, sim.spring_stiffness, x_0, act, sim.dt)
            
            grad[inds] += ForwardDiff.gradient(E, d_1)
        end

        # per-vertex gradient
        for vi = 1:N_p

            inds = SA[2 * vi - 1, 2 * vi]

            x_0 = sim.X[inds]
            v_0 = sim.V[inds]
            d_0 = sim.D[inds]
            d_1 = D_1[inds]
            
            E(d) = vertex_energy(d, d_0, v_0, x_0, sim.dt, sim.M[vi], sim.floor_height, sim.floor_force, sim.floor_friction, sim.g)
            grad[inds] += ForwardDiff.gradient(E, d_1)
        end
    end

    function compute_hessian!(hess, sim::Simulation, D_1, A, a=0.01)
        N_p = size(sim.X)[1] ÷ 2
        N_t = size(sim.ind)[1]
        N_a = size(A)[1]

        #per-triangle gradient
        for ti = 1:N_t

            inds = Vector(index_arr(sim.ind[ti]))

            x_0 = sim.X[inds]
            d_1 = D_1[inds]

            E(d) = triangle_energy(d, sim.A_inv[ti], x_0, a, sim.lambda[ti], sim.mu[ti])

            hess[inds, inds] .+= sim.vol[ti] * ForwardDiff.hessian(E, d_1)
        end

        #Per-actuation gradient
        for ai = 1:N_a
            row = sim.actuators_i[ai]
            col = sim.actuators_j[ai]
            act = A[ai]

            inds = [2 * col - 1, 2 * col, 2*row - 1, 2*row]

            x_0 = sim.X[inds]
            d_1 = D_1[inds]

            E(d) = actuation_energy(d, sim.spring_stiffness, x_0, act, sim.dt)

            hess[inds, inds] .+= ForwardDiff.hessian(E, d_1)
        end

        # per-vertex gradient

        for vi = 1:N_p

            inds = [2 * vi - 1, 2 * vi]

            x_0 = sim.X[inds]
            v_0 = sim.V[inds]
            d_0 = sim.D[inds]
            d_1 = D_1[inds]

            E(d) = vertex_energy(d, d_0, v_0, x_0, sim.dt, sim.M[vi], sim.floor_height, sim.floor_force, sim.floor_friction, sim.g)

            hess[inds, inds] .+= ForwardDiff.hessian(E, d_1)
        end
    end

    function dfda(sim, D_1, A)
        grad = zeros(length(sim.X), length(A))
        N_a = size(A)[1]

        # per-actuation gradient
        for ai = 1:N_a
            row = sim.actuators_i[ai]
            col = sim.actuators_j[ai]
            act = A[ai]

            inds = SA[2 * col - 1, 2 * col, 2*row - 1, 2*row]

            x_0 = sim.X[inds]
            d_1 = D_1[inds]

            function E(d)
                E1(a) = actuation_energy(d, sim.spring_stiffness, x_0, a, sim.dt)
                return ForwardDiff.derivative(E1, act)
            end
            
            grad[inds, ai] += ForwardDiff.gradient(E, d_1)
        end

        grad
    end

    function newton(sim, hessian!, gradient!, x_guess, tol, iterations)
        x = x_guess
        grad = zeros(size(x))
        hess = zeros(length(sim.X), length(sim.X))

        delta = 0

        for _ in 1:iterations
            grad .= 0
            hess .= 0
            gradient!(grad, x)
            hessian!(hess, x)

            delta = hess \ grad
        
            x = x - delta

            if norm(delta) * sim.dt^2 < tol
                break
            end
        end

        return x
    end 
end

