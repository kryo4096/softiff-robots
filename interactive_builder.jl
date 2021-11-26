using GLMakie
using StaticArrays

include("softbody.jl")


function main()
    v = Node(zeros(0, 0))
    ind = [SA[1,2,3]]

    render_ind = Node([1,2,3])

    f = Figure(resolution=(640, 640))
    ax = Axis(f[1, 1], aspect=1, limits=(-0.1, 1.1, -0.1, 1.1))

    deactivate_interaction!(ax, :rectanglezoom)

    display(f)

    running = Node(false)
    
    on(events(f).keyboardbutton) do event
        if event.key == Keyboard.space
            if event.action == Keyboard.press
                running[] = true
            end
        end
    end

    on(events(f).mousebutton, priority=0) do event
        if !running[]
            if event.button == Mouse.left
                if event.action == Mouse.press
                    n = size(v[])[2] + 1

                    pos = Vector(mouseposition(ax.scene))

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

        return Consume(false)
    end

    while !running[]
        sleep(0.1)
    end

    n_vertices = size(v[])[2]
    n_triangles = size(ind)[1]

    X = Float64.(reshape(v[], 2n_vertices))

    lambda = ones(n_triangles) * 4000
    mu = ones(n_triangles) * 4000
    A = map(ti -> edge_mat(X[index_arr(ind[ti])]), 1:n_triangles)
    A_inv = Vector(inv.(A))
    vol = abs.(0.5 * det.(A))

    sim = Simulation(10.0, 0.001, X, zeros(2n_vertices), zeros(2n_vertices), ones(n_vertices), ind, A_inv, vol, lambda, mu)

    clicked_vertex = Node(-1)

    on(events(f).mousebutton, priority=0) do event
        if event.button == Mouse.left
            pos = SVector{2}(mouseposition(ax.scene))
            if event.action == Mouse.press
                for i in 1:n_vertices
                    ind = SA[2i - 1, 2i]

                    if norm(sim.X[ind] + sim.D[ind] - pos) < 0.01
                        clicked_vertex[] = i
                    end
                end
            end

            i = clicked_vertex[]
            if i != -1
                ind = SA[2i - 1, 2i]

                sim.D[ind] = pos - sim.X[ind]
                if event.action == Mouse.release

                    clicked_vertex[] = -1
                end
            end

        end

        return Consume()
    end

    i = 0
    while running[]
        
        D = line_search((grad, D_1) -> compute_gradient_parallel!(grad, sim, D_1), [Pass(1.0, 10, sim.D + sim.V * sim.dt, 1e-3), Pass(0.2, 1000, sim.D, 1e-3)])
        
        if clicked_vertex[] != -1
            pos = SVector{2}(mouseposition(ax.scene))
            ind = SA[2clicked_vertex[] - 1, 2clicked_vertex[]]
            D[ind] = pos - sim.X[ind]
        end

        sim.V .= (D .- sim.D) ./ sim.dt
        sim.D .= D

        if i % 5 == 0
            v[] = reshape(sim.X + sim.D, 2, size(sim.X)[1] ÷ 2)

            sleep(0.01)
        end

        i += 1
    end
end