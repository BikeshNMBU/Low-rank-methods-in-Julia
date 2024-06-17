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


struct DynamicLowRankLayer{T<:AbstractFloat}
    U::Matrix{T}
    S::Matrix{T}
    V::Matrix{T}
    U1::Matrix{T}
    V1::Matrix{T}
    bias::Vector{T}
    #rmax::Int
    #r::Int
    #tol::T
    activation::Function

    function DynamicLowRankLayer(input_size::Int, output_size::Int, rank::Int, activation::Function; dtype::Type{T}=Float32) where T  
        QV, _ = qr(randn(dtype, input_size, rank))
        QU, _ = qr(randn(dtype, output_size, rank))
        U = Matrix(QU)
        V = Matrix(QV)
        S = randn(dtype, rank, rank)
        b = randn(dtype, output_size)
        U1 = randn(dtype, output_size, rank)
        V1 = randn(dtype, input_size, rank)        

        return new{T}(U, S, V, U1, V1, b, activation)  
    end
end

# Specify trainable parameters and non-trainable fields for the custom layer
Flux.@functor DynamicLowRankLayer (U, S, V, bias) #(U1, V1)

function (layer::DynamicLowRankLayer)(x::AbstractMatrix)
#=  r = layer.r
    xU = x * layer.U[:, 1:r]
    xUS = xU * layer.S[1:r, 1:r]
    out = xUS * layer.V[:, 1:r]'  =#
    return layer.activation.(layer.U * layer.S * layer.V' * x.+ layer.bias)
end

function sgd_update!(layer::DynamicLowRankLayer, grads, lr, dlrt_step="basis")
#    r = layer.r

    if dlrt_step == "basis"
        #U0 = layer.U
        #V0 = layer.V
        #S0 = layer.S 
        # Perform K-step
        K = layer.U * layer.S
        dK = grads[layer.U] * layer.S
        K .-= lr * dK
        QU1, _ = qr(K)
        layer.U1 .= Matrix(QU1)

        # Perform L-step
        L = layer.V * layer.S'
        dL = grads[layer.V] * layer.S'
        L .-= lr * dL
        QV1, _ = qr(L)
        layer.V1 .= Matrix(QV1)

        # Project coefficients
        M = layer.U1' * layer.U
        N = layer.V' * layer.V1
        layer.S .= M * layer.S * N

        # Update basis
        layer.U .= layer.U1
        layer.V .= layer.V1

        # Update bias
        layer.bias .-= lr * grads[layer.bias]
    
    elseif dlrt_step == "coefficients"
        # One-step integrate S
        layer.S .-= lr * grads[layer.S]
    else
        error("Wrong step defined: $dlrt_step")
    end
    
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


batch_size = 64
train_loader = DataLoader((train_x, train_y), batchsize=batch_size, shuffle=true)
test_loader = DataLoader((test_x, test_y), batchsize=batch_size)

model = Chain(
    DynamicLowRankLayer(784, 256, 20, Flux.relu),                  
    DynamicLowRankLayer(256, 128, 20, Flux.relu),
    MyDense(128, 10, identity),
    softmax
)


loss(x, y) = crossentropy(model(x), y)
#optimizer = Descent(0.0001)

lr = 0.01  # Learning rate for SGD

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
    for (x, y) in train_loader
        # Initial forward and backward pass for all parameters
        grads = gradient(Flux.params(model)) do
            loss(x, y)
        end
        epoch_train_loss += loss(x, y)
        # Update step for basis (excluding coefficients)
        for layer in model.layers
            if isa(layer, MyDense) || isa(layer, DynamicLowRankLayer)          
                sgd_update!(layer, grads, lr)
            end
        end
        
        # Second forward pass Recompute gradients for coefficients
        grads = gradient(Flux.params(model)) do
            loss(x, y)
        end

        # Update coefficients in DynamicLowRankLayer
        for layer in model.layers
            if isa(layer, DynamicLowRankLayer)
                sgd_update!(layer, grads, lr, "coefficients")
            end
        end
    end

    # Evaluate model accuracy on test set
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