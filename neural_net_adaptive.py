import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Dense layer with dynamic neuron creation
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Calculates output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Add new neurons dynamically
    def add_neurons(self, n_new_neurons):
        # Generate random weights for new neurons with the same number of input connections
        new_weights = 0.10 * np.random.randn(self.n_inputs, n_new_neurons)

        # Append the new weights to the existing weights
        self.weights = np.hstack((self.weights, new_weights))

        # Increase the number of neurons in biases
        self.biases = np.zeros((1, self.weights.shape[1]))

# Assign the number of samples and iterations as integers
samples = range(5)  # Creates a range from 0 to 4 (inclusive)
iterations = 10  # You can change this value as needed

# Create dataset
X, y = spiral_data(samples=len(samples), classes=3)

# Create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Initialize a counter to keep track of the number of samples processed
sample_count = 0

# Iterate through the dataset in batches of 5 samples for the specified number of iterations
for _ in range(iterations):
    for i in samples:
        start_idx = i * 5
        end_idx = (i + 1) * 5

        batch_X = X[start_idx:end_idx]

        # Perform a forward pass with the current batch
        dense1.forward(batch_X)

        # Increment the sample count
        sample_count += len(batch_X)

        # Check if it's time to add new neurons (every 100 samples)
        if sample_count % 100 == 0:
            # Add new neurons (e.g., 2 new neurons)
            dense1.add_neurons(2)

        # Print output of the current batch
        print(f"Output after processing {sample_count} samples:")
        print(dense1.output)

        # Save the output to a CSV file
        output_filename = f"output_iteration_{iterations}_sample_{sample_count}.csv"
        np.savetxt(output_filename, dense1.output, delimiter=',')