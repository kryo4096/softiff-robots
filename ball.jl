using GLMakie
using Zygote


function step(x, v, g, Δt)
    x1 = x + v * Δt
    v1 = v + [0, -g] * Δt

    if x[2] < 0
        v1 +=  [10 * v1[1] * x[2], 100] * Δt
    end   

    return x1, v1
end

function ball_throw(α, v, g, Δt, t_1) 
    pos = zeros(2)
    vel = v * [cos(α), sin(α)]

    for i ∈ 1:length(0:Δt:t_1)
        pos, vel = step(pos, vel, g, Δt)
    end

    return pos[1]
end

function ball_throw_traj(α, v, g, Δt, t_1) 
    pos = zeros(2)
    vel = v * [cos(α), sin(α)]

    ts = 0:Δt:t_1

    n = length(ts)

    traj = zeros(n, 2)

    for i ∈ 1:n
        pos, vel = step(pos, vel, g, Δt)

        traj[i,:] = pos
    end

    return traj
end

function main() 

    Δt = 0.01
    t_1 = 2.0
    v = 1
    g = 1.0

    α = π / 2

    f = Figure()
    Axis(f[1,1])

    traj = Node(zeros(length(0:Δt:t_1), 2))

    display(lines(traj, axis=(;limits=(0.0, v * t_1, -0.2, 0.5 * v^2 / g))))

    dist(x) = ball_throw(x, v, g, Δt, t_1)

    for k ∈ 1:100000
        δα = dist'(α)
        α += δα * 0.0001
        
        traj[] = ball_throw_traj(α, v, g, Δt, t_1)
    
        sleep(0.001)
    end
end

main()