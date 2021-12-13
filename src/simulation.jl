module Simulation
    using JSON
    using GLMakie
    using LinearAlgebra

    include("softbody.jl")
    include("neural_net.jl")

    function read_robot_file(filename)
        s = open(filename, "r") do io
            read(io, String)
        end

        robot = JSON.parse(s)

        indices = [Int64(i) for i in robot["indices"]]
        vertices = [Float64(x) for x in robot["vertices"]]
        actuators = [(Int64(a[1]), Int64(a[2])) for a in robot["actuators"]]

        vertices, indices, actuators
    end

    nestlevel() = 0
    Zygote.@adjoint nestlevel() = nestlevel()+1, _ -> nothing

    function plot_actuators(fig_location, a, vertices)

        actuator_indices = Vector{Int64}()
        for i = 1:length(a)
            append!(actuator_indices, [0, -1, -2, 0, -2, -3] .+ 4 * i)
        end

        mesh!(
            fig_location, 
            lift(vertices) do v
                verts = Matrix{Float64}(undef, 2, 0)

                for (i, j) in a
                    vi = v[:, i]
                    vj = v[:, j]

                    dir = normalize(vj - vi)

                    delta = [-dir[2], dir[1]] * 0.01

                    verts = [verts vi - delta vi + delta vj - delta vj + delta]
                end

                verts
            end,
            actuator_indices,
        )

    end

    function run_simulation(sim::Softbody.Simulation, mv, nn)
        iter = 0

        while iter < 100
            a = get_actuation(nn, [sim.X + sim.D;sim.V])

            function D_1(a) 
                print(nestlevel())
                Softbody.newton(
                    sim,
                    (h, d) -> Softbody.compute_hessian!(h, sim, d, a),
                    (g, d) -> Softbody.compute_gradient!(g, sim, d, a),
                    sim.D,
                    1e-6,
                    2
                )
            end

            function dDdA(a, D1F)
                hess = zeros(length(sim.X), length(sim.X))
                
                d_1 = D1F(a)
                Softbody.compute_hessian!(hess, sim, d_1, a)
                cg(hess, Softbody.dfda(d_1, a))
            end

            eval("Zygote.@adjoint D_1(a)=D_1(a), y->(dDdA(a, D_1)*y,)")

            D = D_1(a)
            sim.V = (D - sim.D) / sim.dt
            sim.D = D

            Zygote.ignore() do
                if iter%10==0
                    mv[] = Softbody.render_verts(sim)
                    sleep(0.001)
                end
            end

            iter += 1
        end

        sim.D
    end

    function run(filename)
        robot = read_robot_file(filename)

        sim = Softbody.create_simulation(robot...)

        v, i, a = robot

        nn = NeuralNet(2*size(v)[1], 32, size(a)[1])

        fig = Figure()

        mv = Observable(Softbody.render_verts(sim))
        running = Observable(true)
        
        on(events(fig).keyboardbutton) do event
            if event.key == Keyboard.escape
                running[] = false
            end
        end

        ax = Axis(fig[1,1], aspect=1.0, limits=(-0.55, 0.55, -0.1, 1.0))
        deactivate_interaction!(ax, :rectanglezoom)

        mesh!(fig[1,1], mv, i)

        plot_actuators(fig[1,1], a, mv)

        display(fig)

        x(nn) = run_simulation(sim, mv, nn)
        Zygote.jacobian(x, nn)
    end

end