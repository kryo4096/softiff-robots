include("builder.jl")
include("simulation.jl")

const PROJECT_LOCATION = @__DIR__ 

function create_robot()
    run_builder()
end

function run_simulation(filename=PROJECT_LOCATION * "/../robots/simple_walker.json")
    Simulation.run(filename)
end

run_simulation()