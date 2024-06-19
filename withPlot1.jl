using XLSX, DataFrames, CSV
using Glob
using Flux
using Random
using Plots
using Dates

Random.seed!(3163)

function read_excel_to_dict(excel_file_path, key_column, value_column, sheet_name="Sheet1")
    xls = XLSX.readxlsx(excel_file_path)
    sheet = xls[sheet_name]
    df = DataFrame(XLSX.eachtablerow(sheet))
    return Dict(df[!, key_column] .=> df[!, value_column])
end

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

parent_folder_path = "D:/ReductionData_W_Start/Train"
X, Y = process_parent_directory(parent_folder_path)
println(size(X), size(Y))

test_folder_path = "D:/ReductionData_W_Start/Test"   
X_test, Y_test = process_parent_directory(test_folder_path)
println(size(X_test), size(Y_test))    

model = Chain(
    Dense(28000, 512, relu),
    Dense(512, 512, relu),
    Dense(512, 256, relu),
    Dense(256, 1)
)

loss(x, y) = Flux.Losses.mse(model(x), y)
optimizer = ADAM(0.01)

Y_reshaped = reshape(Y, 1, size(Y)[1])
Y_test_reshaped = reshape(Y_test, 1, size(Y_test)[1])

function calculate_loss(X, Y_reshaped)
    predictions = model(X)
    return Flux.Losses.mae(predictions, Y_reshaped)
end

function calculate_loss_rmse(X, Y_reshaped)
    predictions = model(X)
    mse = Flux.Losses.mse(predictions, Y_reshaped)
    rmse = sqrt(mse)
    return rmse
end

train_losses, test_losses = Float64[], Float64[]

function print_losses(epoch, X_train, Y_train_reshaped, X_test, Y_test_reshaped)
    train_loss = calculate_loss(X_train, Y_train_reshaped)
    test_loss = calculate_loss(X_test, Y_test_reshaped)
    push!(train_losses, train_loss)
    push!(test_losses, test_loss)
    println("Epoch $epoch, Training loss: $train_loss, Test loss: $test_loss")
end

epochs = 2000
start_time = now()  # Start time for training
for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model), [(X, Y_reshaped)], optimizer)
    print_losses(epoch, X, Y_reshaped, X_test, Y_test_reshaped)
end
end_time = now()  # End time for training
@info "Total training time: $(end_time - start_time)"
test_predictions = model(X_test)
actual_test_labels = Y_test_reshaped

println("Actual Test Labels: ", actual_test_labels)
println("Predicted Values: ", test_predictions)

# Plotting the losses
loss_plot = plot(1:epochs, train_losses, label="Training Loss", xlabel="Epoch", ylabel="Loss", title="Training and Test Losses")
plot!(loss_plot, 1:epochs, test_losses, label="Test Loss", xlabel="Epoch", ylabel="Loss", title="Training and Test Losses")
savefig(loss_plot, "training_test_losses.png")

# Plotting actual vs predicted labels
prediction_plot = scatter(actual_test_labels[:], test_predictions[:], label="Actual vs Predicted", xlabel="Actual Labels", ylabel="Predicted Labels", title="Actual vs Predicted Labels")
savefig(prediction_plot, "actual_vs_predicted.png")
