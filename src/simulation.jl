

module Simulation
    using JSON
    using GLMakie

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

        vertices, indices,  actuators
    end

    function run_simulation(sim, nn, direction, mv, running, max_sim=1000)
        for simulation_step = 1:max_sim
            running[] || break
            Softbody.step!(sim, get_actuation(nn, [sim.X; sim.V; direction[]]))

            Zygote.ignore() do
                if simulation_step%10==0
                    mv[] = Softbody.render_verts(sim)
                    sleep(0.001)
                end
                
            end
        end
        
        return reward(sim, direction[])
    end

    function run(filename)
        robot = read_robot_file(filename)
    
        sim = Softbody.create_simulation(robot...)
    
        v, i, a = robot
    
        nn = NeuralNet(2 * size(v)[1] + 2, 32, size(a)[1])
    
        fig = Figure()
    
        mv = Observable(Softbody.render_verts(sim))
        running = Observable(true)
        direction = Observable([rand(-1:1), rand(-1:1)])
    
        on(events(fig).keyboardbutton) do event
            #direction[] = [0, 0]
            if event.key == Keyboard.escape
                running[] = false
            end
            # if event.key == Keyboard.up
            #     direction[][2] += 1
            # end
            # if event.key == Keyboard.down
            #     direction[][2] -= 1
            # end
            # if event.key == Keyboard.right
            #     direction[][1] += 1
            # end
            # if event.key == Keyboard.left
            #     direction[][1] -= 1
            # end
        end
    
        mesh(fig[1, 1], mv, i)
    
        display(fig)
    
        max_train = 1000
        for training_step = 1:max_train
            running[] || break
            run_simulation(sim, nn, direction, mv, running)
            grad = gradient(run_simulation, sim, nn, direction, mv, running)
            print(grad)
            backpropagate()
        end

        iter = 0
        while running[]
            Softbody.step!(sim, get_actuation(nn, [sim.X; sim.V; direction[]]))
            if iter%10==0
                mv[] = Softbody.render_verts(sim)
                sleep(0.001)
            end
            iter += 1
        end
    end

end