using Flux , MLDatasets, Statistics
using Flux: onehotbatch, crossentropy, onecold, throttle
using Flux.Data: DataLoader
using LinearAlgebra: qr
#using Zygote
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

Flux.@functor MyDense (W, b)  # This specifies that only W and b are trainable parameters

function (layer::MyDense)(x::AbstractMatrix)
    return layer.activation.(layer.W * x .+ layer.b)
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


mutable struct DynamicLowRankLayer{T<:AbstractFloat}
    U::Matrix{T}
    S::Matrix{T}
    V::Matrix{T}
    U1::Matrix{T}
    V1::Matrix{T}
    b::Vector{T}
    activation::Function
    U_momentum::Matrix{T} 
    S_momentum::Matrix{T}
    V_momentum::Matrix{T}
    b_momentum::Vector{T}

    function DynamicLowRankLayer(input_size::Int, output_size::Int, rank::Int, activation::Function; dtype::Type{T}=Float32) where T   #, tol::T=1e-2 
        QV, _ = qr(randn(dtype, input_size, rank))
        QU, _ = qr(randn(dtype, output_size, rank))
        U = Matrix(QU)
        S = randn(dtype, rank, rank)
        V = Matrix(QV)
        U1 = randn(dtype, output_size, rank)
        V1 = randn(dtype, input_size, rank)
        b = randn(dtype, output_size)
        U_m = zeros(size(U))  # Initialize U momentum as zeros
        S_m = zeros(size(S))  # Initialize S momentum as zeros
        V_m = zeros(size(V))  # Initialize V momentum as zeros
        b_m = zeros(size(b))  # Initialize b momentum as zeros        
        return new{T}(U, S, V, U1, V1, b, activation, U_m, S_m, V_m, b_m)  
    end
end

# Specify trainable parameters and non-trainable fields for the custom layer
Flux.@functor DynamicLowRankLayer (U, S, V, b) #(U1, V1)

function (layer::DynamicLowRankLayer)(x::AbstractMatrix)
#=  r = layer.r
    xU = x * layer.U[:, 1:r]
    xUS = xU * layer.S[1:r, 1:r]
    out = xUS * layer.V[:, 1:r]'  =#
    return layer.activation.(layer.U * (layer.S * (layer.V' * x)).+ layer.b)
end

function sgd_update_with_momentum_and_clipping!(layer::DynamicLowRankLayer, grads, lr, momentum, threshold, dlrt_step="basis")
    
    
    if dlrt_step == "basis"
        # Compute gradients and apply gradient clipping
        dU, dV, db = grads[layer.U], grads[layer.V], grads[layer.b]  
        clip_gradient!(dU, threshold)
        clip_gradient!(dV, threshold)
        clip_gradient!(db, threshold)

        # Update momentum
        layer.U_momentum = momentum * layer.U_momentum + (1 - momentum) * dU
        layer.V_momentum = momentum * layer.V_momentum + (1 - momentum) * dV
        layer.b_momentum = momentum * layer.b_momentum + (1 - momentum) * db

        # Perform K-step
        K = layer.U * layer.S
        dK = layer.U_momentum * layer.S
        K .-= lr * dK
        QU1, _ = qr(K)
        layer.U1 .= Matrix(QU1)

        # Perform L-step
        L = layer.V * layer.S'
        dL = layer.V_momentum * layer.S'
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
        layer.b .-= lr * layer.b_momentum
    
    elseif dlrt_step == "coefficients"
        # Compute gradients, apply gradient clipping and update momentum
        dS = grads[layer.S]
        clip_gradient!(dS, threshold)
        layer.S_momentum = momentum * layer.S_momentum + (1 - momentum) * dS
        # One-step integrate S
        layer.S .-= lr * layer.S_momentum
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
train_x = (Flux.flatten(train_x) .- 0.5) ./ 0.5             
test_x = (Flux.flatten(test_x) .- 0.5) ./ 0.5               
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

lr = 0.001  # Learning rate for SGD
momentum = 0.9  # Momentum factor
threshold = 4  # Gradient clipping threshold

accuracy(ymodel, y) = mean(onecold(ymodel) .== onecold(y))

function improved_accuracy(predictions, labels)
    correct = 0
    for i in axes(predictions, 2)  # Iterating over columns
        predicted_label = argmax(predictions[:, i])
        true_label = argmax(labels[:, i])
        correct += (predicted_label == true_label)
    end
    accuracy = correct / size(predictions, 2)  # Total number of predictions
    return accuracy
end

epochs = 200
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
                sgd_update_with_momentum_and_clipping!(layer, grads, lr, momentum, threshold)
            end
        end
        
        # Second forward pass Recompute gradients for coefficients
        layers_with_S = [layer for layer in model.layers if hasfield(typeof(layer), :S)]
        S_params = [getfield(layer, :S) for layer in layers_with_S]

        grads = gradient(Flux.Params(S_params)) do
            loss(x, y)
        end

        # Update coefficients in DynamicLowRankLayer
        for layer in model.layers
            if isa(layer, DynamicLowRankLayer)
                sgd_update_with_momentum_and_clipping!(layer, grads, lr, momentum, threshold, "coefficients")
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
