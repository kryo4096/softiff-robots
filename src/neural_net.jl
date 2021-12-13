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

#NeuralNet(i, hl, o) = NeuralNet(i, hl, o, randn(hl, i), randn(hl), randn(o, hl), randn(o)) Does not work - Random actuations are too far off from equilibrium
NeuralNet(i, hl, o) = NeuralNet(i, hl, o, zeros(hl, i), zeros(hl), zeros(o, hl), ones(o))

function relu(x)
    if x < 0
        return 0
    else
        return x 
    end
end


function get_actuation(nn, input)
    f(x) = relu(x)
    return f.(nn.w2*f.(nn.w1*input .+ nn.b1) .+ nn.b2)
end


function update(nn::NeuralNet, loss)
    dnn = gradient(loss, nn)[1]
    learning_rate = 0.05
    nn.w1 += learning_rate * dnn.w1
    nn.b1 += learning_rate * dnn.b1
    nn.w2 += learning_rate * dnn.w2
    nn.b2 += learning_rate * dnn.b2
end

function reward(pos, dir)
    com = [0, 0]

    for i = 1:2:length(pos)
        com += [pos[i], pos[i+1]]
    end
    com /= 0.5*size(pos)[1]
    return com[1]*dir[1] + com[2]*dir[2]
end
