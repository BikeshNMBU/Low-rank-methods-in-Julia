using Flux , MLDatasets, Statistics
using Flux: onehotbatch, crossentropy, onecold, throttle
using Flux.Data: DataLoader
using LinearAlgebra: qr
using Random
using Dates
using CSV,DataFrames

Random.seed!(3163)  # Set the random seed for reproducibility

struct MyDense
    W::AbstractMatrix
    b::AbstractVector
    activation::Function

    MyDense(in::Int, out::Int, activation::Function) = new(randn(out, in), randn(out), activation)
end

function (layer::MyDense)(x::AbstractMatrix)
    return layer.activation.(layer.W * x .+ layer.b)
end

function sgd_update!(layer::MyDense, grads, lr)
    layer.W .-= lr .* grads[layer.W]
    layer.b .-= lr .* grads[layer.b]
end

Flux.@functor MyDense (W, b)  # This specifies that only W and b are trainable parameters

struct VanillaLowRankLayer
    U::AbstractMatrix  
    S::AbstractMatrix
    V::AbstractMatrix
    b::AbstractVector
    activation::Function

    function VanillaLowRankLayer(input_size::Int, output_size::Int, rank::Int, activation::Function)
        U = randn(output_size, rank)
        S = randn(rank, rank)
        V = randn(input_size, rank)
        b = randn(output_size)

        # Orthonormalization of U and V
        QU, _ = qr(U)
        U = Matrix(QU)
        QV, _ = qr(V)
        V = Matrix(QV)

        return new(U, S, V, b, activation)
    end
end

Flux.@functor VanillaLowRankLayer (U, S, V, b)  # Specifies trainable parameters

function (layer::VanillaLowRankLayer)(x::AbstractMatrix)
    # U*S*V'*x + b
    return layer.activation.(layer.U * layer.S * layer.V' * x .+ layer.b)
end

function sgd_update!(layer::VanillaLowRankLayer, grads, lr)
    layer.U .-= lr .* grads[layer.U]
    layer.S .-= lr .* grads[layer.S]
    layer.V .-= lr .* grads[layer.V]
    layer.b .-= lr .* grads[layer.b]
end




struct VanillaLowRankLayer2{T<:AbstractFloat}
    U::Matrix{T}
    S::Matrix{T}
    V::Matrix{T}
    bias::Vector{T}
    activation::Function

    function VanillaLowRankLayer2(input_size::Int, output_size::Int, rank::Int, activation::Function; dtype::Type{T}=Float32) where T
        QV, _ = qr(randn(dtype, input_size, rank))
        QU, _ = qr(randn(dtype, output_size, rank))
        U = Matrix(QU)
        V = Matrix(QV)
        S = randn(dtype, rank, rank)
        b = randn(dtype, output_size)
        
        return new{T}(U, S, V, b, activation)
    end
end

# Specify trainable parameters for the custom layer
Flux.@functor VanillaLowRankLayer2 (U, S, V, bias)

function (layer::VanillaLowRankLayer2)(x::AbstractMatrix)
    
    return layer.activation.(layer.U * layer.S * layer.V' * x.+ layer.bias)
end

function sgd_update!(layer::VanillaLowRankLayer2, grads, lr)
    layer.U .-= lr .* grads[layer.U]
    layer.S .-= lr .* grads[layer.S]
    layer.V .-= lr .* grads[layer.V]
    layer.bias .-= lr .* grads[layer.bias]
end

# Load and preprocess the MNIST dataset
train_x, train_y = MLDatasets.MNIST(split=:train)[:]
test_x, test_y = MLDatasets.MNIST(split=:test)[:]

# Flatten and normalize data
#train_x = Flux.flatten(train_x) ./ 255
#test_x = Flux.flatten(test_x) ./ 255
train_x = (Flux.flatten(train_x) .- 0.5) ./ 0.5             #./ 255
test_x = (Flux.flatten(test_x) .- 0.5) ./ 0.5               # ./ 255
train_y = Flux.onehotbatch(train_y, 0:9)
test_y = Flux.onehotbatch(test_y, 0:9)
# Flatten and normalize data



batch_size = 64
train_loader = DataLoader((train_x, train_y), batchsize=batch_size, shuffle=true)
test_loader = DataLoader((test_x, test_y), batchsize=batch_size)

#=model = Chain(
    VanillaLowRankLayer2(784, 128, 20, Flux.relu),
    VanillaLowRankLayer2(128, 64, 20, Flux.relu),
    MyDense(64, 10, identity),
    softmax
)=#

model = Chain(
    VanillaLowRankLayer2(784, 256, 20, Flux.relu),
    VanillaLowRankLayer2(256, 128, 20, Flux.relu),
    MyDense(128, 10, identity),
    softmax
)

loss(x, y) = crossentropy(model(x), y)
#optimizer = Descent(0.0001)

lr = 0.001  # Learning rate for SGD

accuracy(ymodel, y) = mean(onecold(ymodel) .== onecold(y))

epochs = 20
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
                sgd_update!(layer, grads, lr)
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