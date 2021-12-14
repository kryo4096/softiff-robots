using LinearAlgebra
using GLMakie
using Zygote
using LinearAlgebra


mutable struct NeuralNet{S <: AbstractFloat,I <: Integer}

    i_size::I
    hl_size::I
    o_size::I

    w1::Matrix{S}
    b1::Vector{S}
    w2::Matrix{S}
    b2::Vector{S}


end

c = 0.2

#NeuralNet(i, hl, o) = NeuralNet(i, hl, o, randn(hl, i), randn(hl), randn(o, hl), randn(o)) #Does not work - Random actuations are too far off from equilibrium
NeuralNet(i, hl, o) = NeuralNet(i, hl, o, c*randn(hl, i), c*randn(hl), c*randn(o, hl), ones(o)) #Does not work - Random actuations are too far off from equilibrium
#NeuralNet(i, hl, o) = NeuralNet(i, hl, o, zeros(hl, i), zeros(hl), zeros(o, hl), ones(o))

function relu(x)
    if x < 0
        return 0
    else
        return x 
    end
end

function id(x)
    return x
end



function get_actuation(nn, input)
    f(x) = id(x)
    return f.(nn.w2*f.(nn.w1*input .+ nn.b1) .+ nn.b2)
end


function update_nn(nn::NeuralNet, dnn )
    learning_rate = 1
    nn.w1 .-= learning_rate * dnn.w1
    nn.b1 .-= learning_rate * dnn.b1
    nn.w2 .-= learning_rate * dnn.w2
    nn.b2 .-= learning_rate * dnn.b2
end
