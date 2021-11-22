using LinearAlgebra:norm_sqr
using LinearAlgebra
using GLMakie
using Zygote
using LinearAlgebra

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

function random_layer(n, m)
    return rand(n, m) * 2 .- 1
end

function main() 

    dt = 0.001
    n = 1000
    g = 1
    learn_rate = 0.05

    inp = 2
    out = 2

    layer1_size = 4
    layer2_size = 4

    training_iterations = 1000

    function throw_physics(x, v, t) 
        return [0, -g]
    end

    A1 = random_layer(layer1_size, inp)
    B1 = random_layer(layer1_size, 1)
    A2 = random_layer(layer2_size, layer1_size)
    B2 = random_layer(layer2_size, 1)
    A3 = random_layer(out, layer2_size)
    B3 = random_layer(out, 1)

    trajectory = Node(zeros(n, 2))
    goal_point = Node(Point2f((0.5, 0.5)))

    p = lines(trajectory, axis=(; limits=(-0.3, 1.3, -0.3, 1.3)))
    scatter!(goal_point)

    display(p)

    for k ∈ 1:100000
        if k >= training_iterations || k % 1 == 0
            goal_point[] = Point2f((rand(), rand()))
        end

        input = [goal_point[][1], goal_point[][2]]

        function loss(A1, B1, A2, B2, A3, B3)

            output = A3 * (A2 * (A1 * input + B1) + B2) + B3

            t, final_pos = integrate(throw_physics, [0.0,0.0], output, dt, n)
    
            Zygote.ignore() do 
                trajectory[] = t
            end
            
            return norm_sqr(final_pos - input)
        end

        loss_val = loss(A1, B1, A2, B2, A3, B3)
        

        if k < training_iterations
            dA1, dB1, dA2, dB2, dA3, dB3 = gradient(loss, A1, B1, A2, B2, A3, B3)

            A1 -= dA1 * learn_rate
            B1 -= dB1 * learn_rate
            A2 -= dA2 * learn_rate
            B2 -= dB2 * learn_rate
            A3 -= dA3 * learn_rate
            B3 -= dB3 * learn_rate

            println("loss = $loss_val")
        else
            
            sleep(0.5)
        end

        if k == training_iterations
            display(A1)
            display(B1)
            display(A2)
            display(B2)
            display(A3)
            display(B3)
        end
    end
end

main()    