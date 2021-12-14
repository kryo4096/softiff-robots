module Simulation
    using JSON
    using GLMakie
    using LinearAlgebra
    using ChainRulesCore
    using IterativeSolvers
    using Flux
    
    import ChainRulesCore.rrule
    import ChainRulesCore.NoTangent
    import ChainRulesCore.Tangent
    import ChainRulesCore.ZeroTangent

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

    function step(a, sim) 
        Softbody.newton(
            sim,
            (h, d) -> Softbody.compute_hessian!(h, sim, d, a),
            (g, d) -> Softbody.compute_gradient!(g, sim, d, a),
            sim.D,
            0.0,
            5
        )
    end
    
    function rrule(::typeof(step), a, sim)
        function dDdA(y)
            hess = zeros(length(sim.X), length(sim.X)) 
            d_1 = step(a, sim)
            Softbody.compute_hessian!(hess, sim, d_1, a)
            g = -hess \ Softbody.dfda(sim, d_1, a)
            (NoTangent(), g'*y, ZeroTangent())
        end

        return step(a, sim), dDdA
    end

    function run_simulation(sim::Softbody.Simulation, mv, nn, max_iter=2000)
        iter = 0

        s = deepcopy(sim)

        while iter < max_iter

            com = [0, 0]
            for ind = 1:length(s.X)÷2
                com += [s.X[2ind-1] + s.D[2ind-1], s.X[2ind] + s.D[2ind]]
            end
            com /= length(s.X)÷2

            com_vec = reshape(repeat(com, length(s.X)÷2), length(s.X))
            #println(com) 
            
            rel_pos = s.X + s.D - com_vec
            a = get_actuation(nn, [rel_pos;s.V])

            

            D = step(a, s)
            s.V = (D - s.D) / s.dt
            s.D = D

            Zygote.ignore() do
                #display(a)
                if iter%10==0
                    mv[] = Softbody.render_verts(s)
                    sleep(0.0005)
                end
            end

            iter += 1

            if iter == max_iter
                println("$com is the Center of Mass for actuation $a")
            end
        end

        

        s.D + s.X
    end

    function run(filename)
        robot = read_robot_file(filename)

        sim = Softbody.create_simulation(robot...)

        v, i, a = robot

        nn = NeuralNet(2*length(v), 10, length(a))

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

        opt = Flux.Optimiser(ClipValue(1e3), Flux.Optimise.ADAM(0.1, (0.9, 0.999)))
        
        for i = 1:30
            function get_com(pos)
                com = [0, 0]
                for ind = 1:length(pos)÷2
                    com += [pos[2ind-1], pos[2ind]]
                end
                com /= length(pos)÷2
                -com[1]
            end

            x(nn) = get_com(run_simulation(sim, mv, nn))

            grad = Zygote.gradient(x, nn)[1]
            update_nn(nn, grad, opt)
            println("Completed iteration $i")
        end

        run_simulation(sim, mv, nn, 100000)

        
    end

end