using GLMakie
using Zygote

function integrate(force, x0, v0, dt, n)
    x = x0
    v = v0
    t = 0.0

    trajectory = zeros(n, 2)

    for i ∈ 1:n
        x += v * dt
        v += force(x, v, t) * dt

        Zygote.ignore() do 
            trajectory[i,:] = x
        end

        t += dt
    end

    return trajectory, x

end


function main() 

    dt = 0.001
    n = 2000
    v = 1
    g = 1.0

    α = π / 2

    f = Figure()
    Axis(f[1,1])

    traj = Node(zeros(n, 2))

    display(lines(traj, axis=(;limits=(0.0, 1.0, 0.0, 1.0))))

    function gravity(x, v, t) 
        f = [0,0]

        if x[2] < 0 
            f += v * x[2] * 100000
        else 
            f += [0, -g]
        end

        return f
    end

    ball_throw(x) = integrate(gravity, [0.0,0.0], v * [cos(x), sin(x)], dt, n)

    da_old = 0
    for k ∈ 1:100000
        traj[], p = ball_throw(α)
        da = gradient(x -> ball_throw(x)[2][1], α)[1]

        α += 0.005 * da / max(abs(da_old - da), 1)

        da_old = da
        
        println("α = $(α / 2π * 360)°")
    end
end

main()    