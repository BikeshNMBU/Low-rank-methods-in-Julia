using XLSX, DataFrames, CSV
using Glob
using Flux
using Random
using Flux.Data: DataLoader
using LinearAlgebra: qr
using Plots
using Dates

Random.seed!(3163)

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


# Reads from an Excel file and returns a dictionary mapping filenames to specified columns
function read_excel_to_dict(excel_file_path, key_column, value_column, sheet_name="Sheet1")
    xls = XLSX.readxlsx(excel_file_path)
    sheet = xls[sheet_name]
    df = DataFrame(XLSX.eachtablerow(sheet))
    return Dict(df[!, key_column] .=> df[!, value_column])
end

# Preprocesses data starting from a specific time
function preprocess_data_from_time(file_path, start_time, num_pairs=14000)
    lines = readlines(file_path)
    data_start_index = findfirst(contains("_DATA"), lines)
    
    times, voltages = Float64[], Float64[]
    
    for line in lines[data_start_index+1:end]
        time, voltage = split(line)
        parsed_time = parse(Float64, time)
        if parsed_time >= start_time
            push!(times, parsed_time)
            push!(voltages, parse(Float64, voltage))
        end
    end
    
    required_length = min(length(times), num_pairs)
    if length(times) < num_pairs
        times = append!(times, zeros(Float64, num_pairs - length(times)))
        voltages = append!(voltages, zeros(Float64, num_pairs - length(voltages)))
    end
    
    interleaved_array = [val for i in 1:required_length for val in (times[i], voltages[i])]
    if length(interleaved_array) < 2 * num_pairs
        interleaved_array = append!(interleaved_array, zeros(Float64, 2 * num_pairs - length(interleaved_array)))
    end
    
    return interleaved_array
end

function create_dataset_for_folder(folder_path)
    start_times_path = joinpath(folder_path, "start_values.XLSX")
    labels_path = joinpath(folder_path, "Extended_Dataset.XLSX")
    
    if !isfile(start_times_path) || !isfile(labels_path)
        println("Required Excel file not found in folder $folder_path")
        return [], []  # Return empty arrays if any Excel file is missing
    end

    start_times_dict = read_excel_to_dict(start_times_path, "Filename", "x_start", "start_values")
    labels_dict = read_excel_to_dict(labels_path, "Filename", "Rate")

    file_paths = glob("*.bka", folder_path)
    features, labels = [], []

    for file_path in file_paths
        filename_with_extension = basename(file_path)
        if haskey(start_times_dict, filename_with_extension) && haskey(labels_dict, filename_with_extension)
            start_time = start_times_dict[filename_with_extension]
            start_time = isa(start_time, Float64) ? start_time : parse(Float64, start_time)  # Check if parse is necessary
            label = labels_dict[filename_with_extension]
            label = isa(label, Float64) ? label : parse(Float64, label)  # Check if parse is necessary

            push!(labels, label)
            feature_vector = preprocess_data_from_time(file_path, start_time)
            push!(features, feature_vector)
        else
            println("Data for file $filename_with_extension not found in both Excel files.")
        end
    end
    
    feature_matrix = hcat(features...)
    return feature_matrix, labels
end

# Processes each subdirectory within the parent directory to create a dataset
function process_parent_directory(parent_folder_path)
    all_features, all_labels = Float64[], Float64[]
    
    subfolders = filter(isdir, readdir(parent_folder_path, join=true))  # Get all subdirectories
    
    for folder in subfolders
        println("Processing folder: $folder")
        X, Y = create_dataset_for_folder(folder)
        if !isempty(X)
            all_features = isempty(all_features) ? X : hcat(all_features, X)
            append!(all_labels, Y)
        end
    end
    
    return all_features, all_labels
end


# Example usage
#parent_folder_path = "D:/ThesisMaterials/JuliaProject/data"
parent_folder_path = "D:/ReductionData_W_Start/Train"
X, Y = process_parent_directory(parent_folder_path)
println(size(X), size(Y))

#test_folder_path = "D:/ThesisMaterials/JuliaProject/test"
test_folder_path = "D:/ReductionData_W_Start/Test" 
X_test, Y_test = process_parent_directory(test_folder_path)
println(size(X_test), size(Y_test))    

#= Example indices to print
indices_to_print = [1,5]  # Just as an example, choose according to your dataset size

for idx in indices_to_print
    println("Features at position $idx: ", X[:, idx])
    println("Label at position $idx: ", Y[idx])
    println()  # Just for an extra newline for readability
end =#

# Define the model
model = Chain(
    DynamicLowRankLayer(28000, 512, 50, relu),
    DynamicLowRankLayer(512, 512, 50, relu),  # Added layer
    DynamicLowRankLayer(512, 256, 50, relu),
    MyDense(256, 1, identity)  # Removed softplus
)


# Define loss function
loss(x, y) = Flux.Losses.mse(model(x), y)

# Optimizer
lr = 0.0001  # Learning rate for SGD
momentum = 0.9  # Momentum factor
threshold = 4  # Gradient clipping threshold

# Ensure Y is reshaped correctly for training
Y_reshaped = reshape(Y, 1, size(Y)[1])
Y_test_reshaped = reshape(Y_test, 1, size(Y_test)[1])

batch_size = 64
train_loader = DataLoader((X, Y_reshaped), batchsize=batch_size, shuffle=true)
test_loader = DataLoader((X_test, Y_test_reshaped), batchsize=batch_size)

# Function to calculate the loss on a given dataset
function calculate_loss(X, Y_reshaped)
    #@assert size(X, 1) == 10000 "Input data shape mismatch during loss calculation"
    predictions = model(X)
    return Flux.Losses.mae(predictions, Y_reshaped)
end

function calculate_loss_rmse(X, Y_reshaped)
    predictions = model(X)
    mse = Flux.mse(predictions, Y_reshaped)
    rmse = sqrt(mse)
    return rmse
end

# Callback function to print training and test loss
function print_losses(epoch, X_train, Y_train_reshaped, X_test, Y_test_reshaped)
    #@assert size(X_train, 1) == 10000 && size(X_test, 1) == 10000 "Data shape mismatch"
    train_loss = calculate_loss(X_train, Y_train_reshaped)
    test_loss = calculate_loss(X_test, Y_test_reshaped)
    println("Epoch $epoch, Training loss: $train_loss, Test loss: $test_loss")
end

# Training loop
epochs = 2000
train_losses, test_losses = Float64[], Float64[]
start_time = now()  # Start time for training
for epoch = 1:epochs
    for (x, y) in train_loader
        # Initial forward and backward pass for all parameters
        grads = gradient(Flux.params(model)) do
            loss(x, y)
        end

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

    train_loss = calculate_loss(X, Y_reshaped)
    test_loss = calculate_loss(X_test, Y_test_reshaped)
    push!(train_losses, train_loss)
    push!(test_losses, test_loss)
    @info "Epoch: $epoch, Training loss: $train_loss, Test loss: $test_loss"
    
end
end_time = now()  # End time for training
@info "Total training time: $(end_time - start_time)"

# Evaluation after the last epoch
test_predictions = model(X_test)
actual_test_labels = Y_test_reshaped

# Print actual and predicted labels
println("Actual Test Labels: ", actual_test_labels)
println("Predicted Values: ", test_predictions)

# Plotting the losses
loss_plot = plot(1:epochs, train_losses, label="Training Loss", xlabel="Epoch", ylabel="Loss", title="Training and Test Losses")
plot!(loss_plot, 1:epochs, test_losses, label="Test Loss", xlabel="Epoch", ylabel="Loss", title="Training and Test Losses")
savefig(loss_plot, "training_test_losses.png")

# Plotting actual vs predicted labels
prediction_plot = scatter(actual_test_labels[:], test_predictions[:], label="Actual vs Predicted", xlabel="Actual Labels", ylabel="Predicted Labels", title="Actual vs Predicted Labels")
savefig(prediction_plot, "actual_vs_predicted.png")
