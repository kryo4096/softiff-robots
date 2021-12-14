using LinearAlgebra
using GLMakie
using Zygote
using LinearAlgebra
using Flux


mutable struct NeuralNet{S <: AbstractFloat,I <: Integer}

    i_size::I
    hl_size::I
    o_size::I

    w1::Matrix{S}
    b1::Vector{S}
    w2::Matrix{S}
    b2::Vector{S}


end


c = 0.5

#NeuralNet(i, hl, o) = NeuralNet(i, hl, o, randn(hl, i), randn(hl), randn(o, hl), randn(o)) #Does not work - Random actuations are too far off from equilibrium
NeuralNet(i, hl, o) = NeuralNet(i, hl, o, c*randn(hl, i), c*randn(hl), c*randn(o, hl), c * randn(o)) 
#NeuralNet(i, hl, o) = NeuralNet(i, hl, o, zeros(hl, i), zeros(hl), zeros(o, hl), ones(o))

function relu(x)
    if x < 0
        return 0
    else
        return x 
    end
end

function leaky_relu(x)
    if x < 0
        return 0.01x
    else
        return x 
    end
end

function tan_h(x)
    return tanh(x) + 1 + 0.01relu(x-1)
end

function id(x)
    return x
end



function get_actuation(nn, input)
    f(x) = leaky_relu(x)
    g(x) = tan_h(x)
    return g.(f.(nn.w2*f.(nn.w1*input .+ nn.b1) .+ nn.b2))
end


function update_nn(nn::NeuralNet, dnn, opt)
    #lr = 0.05
    #nn.w1 -= lr * dnn.w1
    #nn.b1 -= lr * dnn.b1
    #nn.w2 -= lr * dnn.w2
    #nn.b2 -= lr * dnn.b2

    Flux.update!(opt, nn.w1, dnn.w1)
    Flux.update!(opt, nn.b1, dnn.b1)
    Flux.update!(opt, nn.w2, dnn.w2)
    Flux.update!(opt, nn.b2, dnn.b2)
end
