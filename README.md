# What are Partial Differential Equations?
Partial Differential Equations (PDEs) are mathematical equations that involve multiple independent variables, their partial derivatives, and an unknown function. They describe how a quantity changes with respect to several variables and are fundamental in modeling various physical, engineering, and natural phenomena, such as heat conduction, wave propagation, fluid dynamics, electromagnetism, and quantum mechanics.

# Fourier-Neural-Operator
Fourier Neural Operators (FNOs) are a deep learning approach for solving PDEs efficiently by learning function-to-function mappings. Unlike traditional numerical solvers, FNOs operate in the Fourier domain, making them significantly faster and more scalable for high-dimensional PDEs.

Traditional methods for solving PDEs, such as Finite Difference Methods (FDM), Finite Element Methods (FEM), and Spectral Methods, can be computationally expensive and struggle with high-dimensional or complex boundary conditions. Deep learning (DL) offers an alternative that can handle these challenges efficiently.

Fourier Neural Operators (FNOs) outperform CNNs and PINNs in learning PDEs by capturing global dependencies through Fourier transforms, enabling faster convergence and superior scalability. Unlike CNNs, which struggle with long-range interactions, and PINNs, which require complex loss tuning, FNOs learn direct mappings between function spaces, making them highly efficient for high-dimensional problems. Their ability to generalize across varying conditions without retraining makes them ideal for real-time applications like fluid dynamics and climate modeling.

# Operator Learning
Operator learning is a machine learning approach used to learn mappings between functions rather than just input-output pairs. It generalizes neural networks to work with infinite-dimensional spaces, such as solutions to differential equations. 
It learns a transformation $G$ that maps one function to another:

$$G: u(x) \to v(x)$$  

where $u(x)$ is the input function and $v(x)$ is the output function.

<div align="center">
  <img src="./FNO 1.png" alt="Sample of a PDE">
</div>


PDE solutions function as operators that map between function spaces, taking inputs like initial conditions, boundary conditions, and source terms to produce the corresponding solution. Neural Operators are a class of data-driven models designed to be capable of handling and generalizing across different representations, including varying mesh refinements.

# Import Libraries
```python
import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import pandas as pd
```

# Fourier Layer
Since the inputs and outputs of partial differential equations (PDEs) are continuous functions, representing them in Fourier space is often more efficient.

In the spatial domain, convolution corresponds to pointwise multiplication in the Fourier domain. To apply the (global) convolution operator, we first perform a Fourier transform, followed by a linear transformation, and then an inverse Fourier transform.

<div align="center">
  <img src="./fourier_layer.png" alt="Fourier Layers">
</div>

The Fourier layer just consists of three steps:

1. Fourier transform (using FFT) $F$

2. Linear transform on the lower Fourier modes $R$

3. Inverse Fourier transform $F^(-1)$


The SpectralConv1d (or Fourier layer) is a specialized layer designed to perform convolution in the Fourier domain rather than in the spatial domain.

```python

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer: It applies FFT, performs a learnable transformation in the frequency domain,
        and then applies the inverse FFT to bring the data back to the spatial domain.
        """

        # Number of input and output channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to keep (frequency truncation)
        self.modes1 = modes1

        # Scaling factor to normalize weight initialization
        self.scale = (1 / (in_channels * out_channels))

        # Learnable weight parameters (complex numbers) for the Fourier modes
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Define a function for complex multiplication in the Fourier domain
    def compl_mul1d(self, input, weights):
        """
        Performs complex multiplication between input Fourier coefficients and learnable weights.
        Uses Einstein summation notation for efficient tensor multiplication.

        Args:
            input: Fourier-transformed input tensor of shape (batch, in_channels, frequency modes)
            weights: Learnable weight tensor of shape (in_channels, out_channels, frequency modes)

        Returns:
            Tensor of shape (batch, out_channels, frequency modes)
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        """
        Forward pass of the Spectral Convolution Layer.
        1. Apply FFT to transform input to frequency domain.
        2. Multiply the first 'modes1' Fourier coefficients with learnable weights.
        3. Apply inverse FFT to transform back to spatial domain.

        Args:
            x: Input tensor of shape (batch_size, in_channels, number of grid points)

        Returns:
            Transformed tensor of the same shape as input.
        """

        batchsize = x.shape[0]  # Extract batch size

        # Step 1: Compute the Fourier Transform (Real FFT) to get frequency components
        x_ft = torch.fft.rfft(x)

        # Step 2: Create an output tensor for storing transformed Fourier coefficients
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)

        # Step 3: Apply the learnable spectral convolution on the first 'modes1' Fourier coefficients
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Step 4: Perform the Inverse Fourier Transform to return to the spatial domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))

        return x
```
# Allen - Cahn Equation
The Allen-Cahn equation is a reaction-diffusion partial differential equation (PDE) used to model phase separation in multi-phase systems.

$$u_t = \Delta u - ϵ^2 u (u^2 - 1),\quad u\in\mathbb{R}×\mathbb{R_{>0}}$$

# Data Exploration and Visualization
```
from google.colab import drive
drive.mount('/content/drive')
```
```
torch.manual_seed(0)
np.random.seed(0)
```
```
# Number of training samples to be used
n_train = 100

# Load the input data from a NumPy file and convert it to a PyTorch tensor
# This file contains the initial conditions and spatial coordinates
x_data = torch.from_numpy(np.load("/content/drive/MyDrive/1D_Allen-Cahn/AC_data_input.npy")).type(torch.float32)

# Load the output data from a NumPy file and convert it to a PyTorch tensor
# This file contains the solution of the equation at a later timestep
y_data = torch.from_numpy(np.load("/content/drive/MyDrive/1D_Allen-Cahn/AC_data_output.npy")).type(torch.float32)

# Swap the first and second channels in the input tensor
temporary_tensor = torch.clone(x_data[:, :, 0])
x_data[:, :, 0] = x_data[:, :, 1]
x_data[:, :, 1] = temporary_tensor

# Split the dataset into training and testing sets
input_function_train = x_data[:n_train, :]
output_function_train = y_data[:n_train, :]

input_function_test = x_data[n_train:, :]
output_function_test = y_data[n_train:, :]

batch_size = 10

# Create a DataLoader for training data
training_set = DataLoader(TensorDataset(input_function_train, output_function_train), batch_size=batch_size, shuffle=True)

# Create a DataLoader for testing data
testing_set = DataLoader(TensorDataset(input_function_test, output_function_test), batch_size=batch_size, shuffle=False)
```
```
# Load the datasets
input_data = x_data
output_data = y_data

# Print dataset shapes
print(f"Input Data Shape: {input_data.shape}")  # Example: (num_samples, time_steps, features)
print(f"Output Data Shape: {output_data.shape}")  # Example: (num_samples, time_steps)

# Check if there are time steps
num_samples = input_data.shape[0]
time_steps = input_data.shape[1] if len(input_data.shape) > 1 else 1
num_features = input_data.shape[2] if len(input_data.shape) > 2 else 1

print(f"Number of Samples: {num_samples}")
print(f"Time Steps: {time_steps}")
print(f"Number of Features in Input: {num_features}")

# Visualize a few sample inputs and outputs
sample_idx = np.random.randint(0, num_samples)  # Select a random sample

plt.figure(figsize=(12, 5))

# Plot Input Data
plt.subplot(1, 2, 1)
for feature in range(num_features):
    if feature == 1:  # Skip Feature 2 (index 1)
        continue
    plt.plot(range(time_steps), input_data[sample_idx, :, feature], label=f'Feature {feature+1}')
plt.title(f'Input Data Sample {sample_idx}')
plt.xlabel('Time Steps')
plt.ylabel('Feature Values')
plt.legend()
plt.grid(True, linestyle=':')


# Plot Output Data
plt.subplot(1, 2, 2)
plt.plot(range(time_steps), output_data[sample_idx, :], label='Output Data', color='r')
plt.title(f'Output Data Sample {sample_idx}')
plt.xlabel('Time Steps')
plt.ylabel('Output Values')
plt.legend()
plt.grid(True, linestyle=':')

plt.tight_layout()
plt.show()
```
Input Data Shape: torch.Size([1000, 1001, 2])
Output Data Shape: torch.Size([1000, 1001])
Number of Samples: 1000
Time Steps: 1001
Number of Features in Input: 2

<div align="center">
  <img src="./FNO 1.png" alt="Sample of a PDE">
</div>


```

```


# FNO1d Model

``` 
# Define the Fourier Neural Operator (FNO) for 1D problems
class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall structure of the Fourier Neural Operator (FNO):
        1. Lift the input to a higher-dimensional space using `self.linear_p`.
        2. Apply multiple layers of Fourier-based convolution (`SpectralConv1d`).
        3. Apply linear transformations (`Conv1d`) after each Fourier layer.
        4. Project the output to the desired final dimension (`self.linear_q` and `self.output_layer`).

        Input:  (batch_size, x=s, c=2) -> initial condition and spatial coordinate (u0(x), x)
        Output: (batch_size, x=s, c=1) -> solution at a later time step
        """

        # Number of Fourier modes to keep (controls how much frequency information is retained)
        self.modes1 = modes

        # Number of channels (width of the network)
        self.width = width

        # Padding: Used to extend the domain if the input is not periodic
        self.padding = 1

        # Initial linear layer to lift the input dimension
        self.linear_p = nn.Linear(2, self.width)

        # Define 3 Fourier convolution layers (SpectralConv1d)
        # Each layer performs spectral convolution to capture global dependencies
        self.spect1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect3 = SpectralConv1d(self.width, self.width, self.modes1)

        # Define 1x1 convolution layers to refine local features after each Fourier layer
        self.lin0 = nn.Conv1d(self.width, self.width, 1)  # First linear transformation
        self.lin1 = nn.Conv1d(self.width, self.width, 1)  # Second linear transformation
        self.lin2 = nn.Conv1d(self.width, self.width, 1)  # Third linear transformation

        # Final linear layers to project to the output dimension
        self.linear_q = nn.Linear(self.width, 32)   # Reduce width to 32
        self.output_layer = nn.Linear(32, 1)        # Reduce final output to 1 channel

        # Activation function (Tanh) for non-linearity
        self.activation = torch.nn.Tanh()

    # Function to apply a Fourier convolution + pointwise linear transformation
    def fourier_layer(self, x, spectral_layer, conv_layer):
        # Apply the spectral convolution and add the corresponding Conv1D layer
        return self.activation(spectral_layer(x) + conv_layer(x))

    # Function to apply a linear transformation followed by activation
    def linear_layer(self, x, linear_transformation):
        return self.activation(linear_transformation(x))

    def forward(self, x):
        """
        Steps:
        1. Lift the input to a higher-dimensional space.
        2. Apply spectral convolution layers followed by linear transformations.
        3. Project the output to the final shape.
        """

        # Step 1: Lift input dimensions using the linear layer
        x = self.linear_p(x)

        # Permute dimensions to match (batch, channels, grid_points) for Conv1D
        x = x.permute(0, 2, 1)  # Shape: (batch_size, width, grid_points)

        # Step 2: Apply three Fourier convolution layers with linear corrections
        x = self.fourier_layer(x, self.spect1, self.lin0)
        x = self.fourier_layer(x, self.spect2, self.lin1)
        x = self.fourier_layer(x, self.spect3, self.lin2)

        # Permute back to (batch, grid_points, width) for the final linear layers
        x = x.permute(0, 2, 1)

        # Step 3: Apply final linear transformations
        x = self.linear_layer(x, self.linear_q)  # Reduce dimension to 32
        x = self.output_layer(x)  # Final projection to 1D output

        return x  
```


















