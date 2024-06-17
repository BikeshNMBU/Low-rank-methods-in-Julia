using CSV, DataFrames, Plots

# Load metrics from disk
train_losses = CSV.read("train_losses.csv", DataFrame).train_losses
test_losses = CSV.read("test_losses.csv", DataFrame).test_losses
train_accuracies = CSV.read("train_accuracies.csv", DataFrame).train_accuracies
test_accuracies = CSV.read("test_accuracies.csv", DataFrame).test_accuracies

# Plotting accuracy and loss
epochs_range = 1:length(train_losses)

# Create a plot with losses
p1 = plot(epochs_range, train_losses, label="Train Loss", xlabel="Epoch", ylabel="Loss", title="Training and Test Loss", legend=:topright)
plot!(p1, epochs_range, test_losses, label="Test Loss")

# Create a second plot for accuracies
p2 = plot(epochs_range, train_accuracies, label="Train Accuracy", xlabel="Epoch", ylabel="Accuracy", title="Training and Test Accuracy", legend=:bottomright)
plot!(p2, epochs_range, test_accuracies, label="Test Accuracy")

# Combine the plots into one layout and make it taller
combined_plot = plot(p1, p2, layout = (2, 1), size=(800, 800))

# Save the combined plot
savefig(combined_plot, "loss_accuracy_plot.png")
