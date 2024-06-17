using Flux , MLDatasets, Statistics
using Flux: onehotbatch, crossentropy, onecold, throttle
using Flux.Data: DataLoader
using LinearAlgebra: qr
using Random
using Dates
using CSV,DataFrames

Random.seed!(3163)  # Set the random seed for reproducibility

function clip_gradient!(grad, threshold)
    grad_norm = sqrt(sum(grad .^ 2))
    if grad_norm > threshold
        grad .*= threshold / grad_norm
    end
end

mutable struct MyDense
    W::AbstractMatrix
    b::AbstractVector
    activation::Function
    W_momentum::AbstractMatrix  
    b_momentum::AbstractVector  

    #MyDense(in::Int, out::Int, activation::Function) = new(randn(out, in), randn(out), activation)
    function MyDense(in::Int, out::Int, activation::Function)
        W = randn(out, in)
        b = randn(out)
        W_m = zeros(size(W))  # Initialize weight momentum as zeros
        b_m = zeros(size(b))  # Initialize bias momentum as zeros
        new(W, b, activation, W_m, b_m)
    end
end

function (layer::MyDense)(x::AbstractMatrix)
    return layer.activation.(layer.W * x .+ layer.b)
end

function sgd_update!(layer::MyDense, grads, lr)
    layer.W .-= lr .* grads[layer.W]
    layer.b .-= lr .* grads[layer.b]
end

function sgd_update_with_momentum_and_clipping!(layer::MyDense, grads, lr, momentum, threshold)
    
    # Compute gradients and apply gradient clipping
    dW, db = grads[layer.W], grads[layer.b]
    clip_gradient!(dW, threshold)
    clip_gradient!(db, threshold)

    # Update momentum
    layer.W_momentum = momentum * layer.W_momentum + (1 - momentum) * dW
    layer.b_momentum = momentum * layer.b_momentum + (1 - momentum) * db

    # Apply updates
    layer.W -= lr * layer.W_momentum
    layer.b -= lr * layer.b_momentum
end

Flux.@functor MyDense (W, b)  # This specifies that only W and b are trainable parameters


mutable struct VanillaLowRankLayer2{T<:AbstractFloat}
    U::Matrix{T}
    S::Matrix{T}
    V::Matrix{T}
    b::Vector{T}
    activation::Function
    U_momentum::Matrix{T} 
    S_momentum::Matrix{T}
    V_momentum::Matrix{T}
    b_momentum::Vector{T}

    function VanillaLowRankLayer2(input_size::Int, output_size::Int, rank::Int, activation::Function; dtype::Type{T}=Float32) where T
        QV, _ = qr(randn(dtype, input_size, rank))
        QU, _ = qr(randn(dtype, output_size, rank))
        U = Matrix(QU)
        S = randn(dtype, rank, rank)
        V = Matrix(QV)
        b = randn(dtype, output_size)
        U_m = zeros(size(U))  # Initialize U momentum as zeros
        S_m = zeros(size(S))  # Initialize S momentum as zeros
        V_m = zeros(size(V))  # Initialize V momentum as zeros
        b_m = zeros(size(b))  # Initialize b momentum as zeros
        return new{T}(U, S, V, b, activation, U_m, S_m, V_m, b_m)
    end
end

# Specify trainable parameters for the custom layer
Flux.@functor VanillaLowRankLayer2 (U, S, V, b)

function (layer::VanillaLowRankLayer2)(x::AbstractMatrix)
    
    return layer.activation.(layer.U * (layer.S * (layer.V' * x)).+ layer.b)
end

function sgd_update!(layer::VanillaLowRankLayer2, grads, lr)
    layer.U .-= lr .* grads[layer.U]
    layer.S .-= lr .* grads[layer.S]
    layer.V .-= lr .* grads[layer.V]
    layer.b .-= lr .* grads[layer.b]
end

function sgd_update_with_momentum_and_clipping!(layer::VanillaLowRankLayer2, grads, lr, momentum, threshold)
    
    # Compute gradients and apply gradient clipping
    dU, dS, dV, db = grads[layer.U], grads[layer.S], grads[layer.V], grads[layer.b]  
    clip_gradient!(dU, threshold)
    clip_gradient!(dS, threshold)
    clip_gradient!(dV, threshold)
    clip_gradient!(db, threshold)

    # Update momentum
    layer.U_momentum = momentum * layer.U_momentum + (1 - momentum) * dU
    layer.S_momentum = momentum * layer.S_momentum + (1 - momentum) * dS
    layer.V_momentum = momentum * layer.V_momentum + (1 - momentum) * dV
    layer.b_momentum = momentum * layer.b_momentum + (1 - momentum) * db

    # Apply updates
    layer.U -= lr * layer.U_momentum
    layer.S -= lr * layer.S_momentum
    layer.V -= lr * layer.V_momentum
    layer.b -= lr * layer.b_momentum
end


# Load and preprocess the MNIST dataset
train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

# Flatten and normalize data
train_x = (Flux.flatten(train_x) .- 0.5) ./ 0.5             #./ 255
test_x = (Flux.flatten(test_x) .- 0.5) ./ 0.5               # ./ 255
train_y = Flux.onehotbatch(train_y, 0:9)
test_y = Flux.onehotbatch(test_y, 0:9)


batch_size = 64
train_loader = DataLoader((train_x, train_y), batchsize=batch_size, shuffle=true)
test_loader = DataLoader((test_x, test_y), batchsize=batch_size)

model = Chain(
    VanillaLowRankLayer2(784, 256, 20, Flux.relu),                  #VanillaLowRankLayer2
    VanillaLowRankLayer2(256, 128, 20, Flux.relu),
    MyDense(128, 10, identity),
    softmax
)


loss(x, y) = crossentropy(model(x), y)
#optimizer = Descent(0.0001)

lr = 0.001  # Learning rate for SGD
momentum = 0.9  # Momentum factor
threshold = 4  # Gradient clipping threshold

accuracy(ymodel, y) = mean(onecold(ymodel) .== onecold(y))

epochs = 200

# Arrays to store the accuracy and loss values
train_losses = Float64[]
test_losses = Float64[]
train_accuracies = Float64[]
test_accuracies = Float64[]
start_time = now()  # Start time for training
for epoch = 1:epochs
    epoch_train_loss = 0.0
    for (i, (x, y)) in enumerate(train_loader)
        grads = gradient(Flux.params(model)) do
            loss(x, y)
        end
        epoch_train_loss += loss(x, y)
        for layer in model.layers
            if isa(layer, MyDense) || isa(layer, VanillaLowRankLayer2)
                sgd_update_with_momentum_and_clipping!(layer, grads, lr, momentum, threshold)
            end
        end
    end
    
    #ymodel = model(test_x)
    #@info "Epoch: $epoch, Test accuracy: $(accuracy(ymodel, test_y))"
    ymodel_train = model(train_x)
    train_acc = accuracy(ymodel_train, train_y)
    push!(train_accuracies, train_acc)
    
    ymodel_test = model(test_x)
    test_loss = loss(test_x, test_y)  # Calculate test loss
    test_acc = accuracy(ymodel_test, test_y)
    
    push!(train_losses, epoch_train_loss / length(train_loader))
    push!(test_losses, test_loss)
    push!(test_accuracies, test_acc)
    
    @info "Epoch: $epoch, Train accuracy: $train_acc, Test accuracy: $test_acc, Test loss: $test_loss"
end 

end_time = now()  # End time for training
@info "Total training time: $(end_time - start_time)"

# Save metrics to disk
CSV.write("train_losses.csv", DataFrame(train_losses=train_losses))
CSV.write("test_losses.csv", DataFrame(test_losses=test_losses))
CSV.write("train_accuracies.csv", DataFrame(train_accuracies=train_accuracies))
CSV.write("test_accuracies.csv", DataFrame(test_accuracies=test_accuracies))
