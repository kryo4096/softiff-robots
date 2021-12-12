

module Simulation
    using JSON
    using GLMakie

    include("softbody.jl")

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

    function run(filename)
        robot = read_robot_file(filename)

        sim = Softbody.create_simulation(robot...)

        v, i, a = robot

        fig = Figure()

        mv = Observable(Softbody.render_verts(sim))
        running = Observable(true)
        
        on(events(fig).keyboardbutton) do event
            if event.key == Keyboard.escape
                running[] = false
            end
        end

        mesh(fig[1,1], mv, i)

        display(fig)

        i = 0
        while running[]
            Softbody.step!(sim, ones(size(sim.a)))

            if i%10==0
                mv[] = Softbody.render_verts(sim)
                sleep(0.001)
            end

            i += 1
        end
    end

end