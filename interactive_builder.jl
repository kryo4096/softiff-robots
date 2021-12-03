using GLMakie
using GLMakie.GLFW
using StaticArrays

include("softbody.jl")

function main()
    v = Node(zeros(0, 0))
    ind = [SA[1,2,3]]

    render_ind = Node([1,2,3])

    f = Figure(resolution=(640, 640))


    ax = Axis(f[1, 1], aspect=1, limits=(-0.1, 1.1, -0.1, 1.1))

    deactivate_interaction!(ax, :rectanglezoom)

    glfw_window = GLMakie.to_native(display(f))

    running = Node(false)
    
    on(events(f).keyboardbutton) do event
        if event.key == Keyboard.space
            if event.action == Keyboard.press
                running[] = true
            end
        end

        if event.key == Keyboard.q
            running[] = false
        end
    end

    on(events(f).mousebutton, priority=0) do event
        if !running[]
            if event.button == Mouse.left
                if event.action == Mouse.press
                    n = size(v[])[2] + 1

                    pos = Vector(mouseposition(ax.scene))

                    new_vertex = true

                    for i in 1:size(v[])[2]
                        if norm(v[][:,i] - pos) < 0.03
                            ni = SA[i, ind[size(ind)[1]][1], ind[size(ind)[1]][2]]

                            push!(ind, ni)
                            render_ind[] = Vector(vcat(ind...))
                            new_vertex = false
                        end
                    end

                    if new_vertex

                        if n > 3
                            push!(ind, SA[n, n - 1, n - 2])
                            render_ind.val = Vector(vcat(ind...))
                        end

                        if n == 1
                            v = Node(reshape(pos, 2, 1))
                            scatter!(f[1,1], v)
                        else
                            v[] = [v[] pos]
                        end

                        if n == 3
                            mesh!(f[1,1], v, render_ind)
                        end

                    end
                end
            end
        end

        return Consume(false)
    end

    while !running[]
        sleep(0.1)
    end

    n_vertices = size(v[])[2]
    n_triangles = size(ind)[1]

    X = Float64.(reshape(v[], 2n_vertices))

    lambda = ones(n_triangles) * 1e5
    mu = ones(n_triangles) * 1e5
    A = map(ti -> edge_mat(X[index_arr(ind[ti])]), 1:n_triangles)
    A_inv = Vector(inv.(A))
    vol = abs.(0.5 * det.(A))

    sim = Simulation(10.0, 1e5, 0.0, 100.,  0.001, X, zeros(2n_vertices), zeros(2n_vertices), ones(n_vertices), ind, A_inv, vol, lambda, mu)

    clicked_vertex = Node(-1)

    on(events(f).mousebutton, priority=0) do event
        if event.button == Mouse.left
            pos = SVector{2}(mouseposition(ax.scene))
            if event.action == Mouse.press
                for i in 1:n_vertices
                    ind = SA[2i - 1, 2i]

                    if norm(sim.X[ind] + sim.D[ind] - pos) < 0.03
                        clicked_vertex[] = i
                    end
                end
            end

            i = clicked_vertex[]
            if i != -1
                if event.action == Mouse.release
                    clicked_vertex[] = -1
                end
            end

        end

        return Consume()
    end

    hess = spzeros(size(sim.X)[1], size(sim.X)[1])
    init_hessian!(hess, sim, 1.0)

    display(hess)

    i = 0
    while running[]
        hess_f(hess, D_1) = compute_hessian!(hess, sim, D_1)
        grad_f(grad, D_1) = compute_gradient!(grad, sim, D_1)

        # D = line_search(grad_f, [Pass(1.0, 10, sim.D + sim.V * sim.dt, 1e-3), Pass(0.1, 1000, sim.D, 1e-3)])
        D = newton!(hess, hess_f, grad_f, sim.D + sim.V * sim.dt, 1e-5)

        if clicked_vertex[] != -1
            pos = SVector{2}(mouseposition(ax.scene))
            ind = SA[2clicked_vertex[] - 1, 2clicked_vertex[]]
            target = pos - sim.X[ind]

            D[ind] .+= normalize(target - D[ind]) .* sim.dt
        end

        sim.V .= (D .- sim.D) ./ sim.dt
        sim.D .= D

        if i % 1 == 0
            v[] = reshape(sim.X + sim.D, 2, size(sim.X)[1] ÷ 2)
        end

        sleep(0.0001)
        i += 1
    end

    GLFW.SetWindowShouldClose(glfw_window, true)
end