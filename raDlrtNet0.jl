using Flux , MLDatasets, Statistics
using Flux: onehotbatch, crossentropy, onecold, throttle
using Flux.Data: DataLoader
using LinearAlgebra: qr, svd, Diagonal
using Random
using Dates
using CSV,DataFrames

Random.seed!(3163)  # Set the random seed for reproducibility


struct MyDense
    W::AbstractMatrix
    b::AbstractVector
    activation::Function

    #MyDense(in::Int, out::Int, activation::Function) = new(randn(out, in), randn(out), activation)
    function MyDense(in::Int, out::Int, activation::Function)
        W = randn(out, in)
        b = randn(out)
        
        new(W, b, activation)
    end
end
function (layer::MyDense)(x::AbstractMatrix)
    return layer.activation.(layer.W * x .+ layer.b)
end

function sgd_update!(layer::MyDense, grads, lr)
    layer.W .-= lr .* grads[layer.W]
    layer.b .-= lr .* grads[layer.b]
end

Flux.@functor MyDense (W, b)  # This specifies that only W and b are trainable parameters



mutable struct RADynamicLowRankLayer{T<:AbstractFloat}
    U::Matrix{T}
    S::Matrix{T}
    V::Matrix{T}
    U1::Matrix{T}   # Used for temporary storage during updates
    V1::Matrix{T}   # Used for temporary storage during updates
    bias::Vector{T}
    rMax::Int       # Maximum rank
    rCurrent::Int   # Current rank
    tol::T          # Tolerance for truncation
    activation::Function

    function RADynamicLowRankLayer(input_size::Int, output_size::Int, rank::Int, tol::T, activation::Function; dtype::Type{T}=Float32) where T
        QU, _ = qr(randn(dtype, output_size, 2 * rank))
        QV, _ = qr(randn(dtype, input_size, 2 * rank))        
        U = Matrix(QU)
        V = Matrix(QV)
        S = randn(dtype, 2 * rank, 2 * rank)
        b = randn(dtype, output_size)    
        # Initialize U1 and V1 with similar shape and type as U and V
        U1 = randn(dtype, output_size, 2 * rank)
        V1 = randn(dtype, input_size, 2 * rank)
        
        rMax = rank
        rCurrent = Int(ceil(rank * 0.5))  # Initializing at a lower rank based on initial compression
        U_m = zeros(size(U))  # Initialize U momentum as zeros
        S_m = zeros(size(S))  # Initialize S momentum as zeros
        V_m = zeros(size(V))  # Initialize V momentum as zeros
        b_m = zeros(size(b))  # Initialize b momentum as zeros        
        new{T}(U, S, V, U1, V1, b, rMax, rCurrent, tol, activation)
    end
    
end

# Make U, S, V, bias trainable, while others are not
Flux.@functor RADynamicLowRankLayer (U, S, V, bias)

function (layer::RADynamicLowRankLayer)(x::AbstractMatrix)
    r = layer.rCurrent
    return layer.activation.(layer.U[:, 1:r] * layer.S[1:r, 1:r] * layer.V[:, 1:r]' * x .+ layer.bias)
end

function sgd_update!(layer::RADynamicLowRankLayer, grads, lr, dlrt_step="basis")
    
    r = layer.rCurrent
    if dlrt_step == "basis"
        U0 = layer.U[:, 1:r]
        V0 = layer.V[:, 1:r]
        S0 = layer.S[1:r, 1:r]

        # Perform K-step
        K = layer.U[:, 1:r] * layer.S[1:r, 1:r]
        dK = grads[layer.U][:, 1:r] * S0
	    #K .-= lr * dK
        #K_ext = [U0 K]
        K_ext = [U0 dK] # Extended K for QR decomposition
        #layer.U1[:, 1:2*r], _ = qr(K_ext)
        QK, _ = qr(K_ext)
        layer.U1[:, 1:2*r] .= QK[:, 1:2*r]  # Use broadcasting with .=
        
        # Perform L-step
        L = layer.V[:, 1:r] * layer.S[1:r, 1:r]'
        dL = grads[layer.V][:, 1:r] * S0'
	    #L .-= lr * dL
        #L_ext = [V0 L]
        L_ext = [V0 dL] # Extended L for QR decomposition
        #layer.V1[:, 1:2*r], _ = qr(L_ext)
        QL, _ = qr(L_ext)
        layer.V1[:, 1:2*r] .= QL[:, 1:2*r]  # Use broadcasting with .=


        # Update U and V with the results from QR decomposition
        layer.U[:, 1:2*r] .= layer.U1[:, 1:2*r]
        layer.V[:, 1:2*r] .= layer.V1[:, 1:2*r]
        
        # Calculate M and N for the projection of S
        M = layer.U1[:, 1:2*r]' * U0
        N = V0' * layer.V1[:, 1:2*r]

        # Update S using M and N
        layer.S[1:2*r, 1:2*r] .= M * S0 * N

        layer.rCurrent = 2*layer.rCurrent

        # Update bias
        layer.bias .-= lr * grads[layer.bias]

    elseif dlrt_step == "coefficients"
        # Integrate S and truncate if necessary
        layer.S[1:layer.rCurrent, 1:layer.rCurrent] .-= lr * grads[layer.S][1:layer.rCurrent, 1:layer.rCurrent]
        truncation_logic!(layer)
    else
        error("Wrong step defined: $dlrt_step")
    end
end


function truncation_logic!(layer::RADynamicLowRankLayer{T}) where T
    # Perform SVD on the current S matrix up to the current rank
    r = layer.rCurrent
    U, sigma, Vt = svd(layer.S[1:r, 1:r])
    
    # Determine the new rank based on tolerance
    tol = layer.tol * maximum(abs.(sigma))
    new_r = findfirst(x -> x < tol, abs.(sigma))

    # If all singular values are above the tolerance, keep the rank as is; otherwise, update it
    new_r = isnothing(new_r) ? r : new_r - 1

    # Cap the new rank at rMax and ensure it does not fall below 2
    new_r = min(max(new_r, 2), layer.rMax)

    # Reassemble the truncated S, U, and V matrices
    layer.S[1:new_r, 1:new_r] .= Diagonal(sigma[1:new_r])
    layer.U[:, 1:new_r] .= layer.U[:, 1:r] * U[:, 1:new_r]
    layer.V[:, 1:new_r] .= layer.V[:, 1:r] * Vt[1:new_r, :]'

    # Update the current rank
    layer.rCurrent = new_r

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
    RADynamicLowRankLayer(784, 256, 40, Float32(5e-3), Flux.relu),                  
    RADynamicLowRankLayer(256, 128, 40, Float32(5e-3), Flux.relu),
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
            if isa(layer, MyDense) || isa(layer, RADynamicLowRankLayer)          
                sgd_update!(layer, grads, lr)
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
            if isa(layer, RADynamicLowRankLayer)
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