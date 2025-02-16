

<div align="center">
  <img src="./models_viz/figs/new_FNO.png" alt="new_FNO.png">
</div>

<div align="center">
  <img src="./models_viz/figs/cnn_model.png" alt="cnn_model.png">
</div>

# Fourier Neural Operators (FNOs) for Learning the 1D Allen Cahn Equation

In this project, we focus on learning the **1D Allen–Cahn equation** using a **Fourier neural operator (FNO) model**. The Allen–Cahn equation is a reaction-diffusion PDE that describes phase separation in multi-phase systems:

$$
u_t = \Delta u - ϵ^2 u (u^2 - 1),\quad u\in\mathbb{R}×\mathbb{R_{>0}}
$$

We demonstrate that the FNO outperforms conventional deep learning approaches, specifically a CNN-based model, in terms of accuracy and generalization. By leveraging FNO’s ability to learn mappings in function space, we achieve superior resolution-invariance and improved performance over mesh-dependent deep learning methods.


## Table of Contents

- [Fourier-Neural-Operator](#fourier-neural-operator)
- [Operator Learning](#operator-learning)
- [Import Libraries](#import-libraries)
- [Fourier Layer](#fourier-layer)
- [Allen-Cahn Equation](#allen---cahn-equation)
- [Data Exploration and Visualization](#data-exploration-and-visualization)
- [FNO1d Model](#fno1d-model)
- [Training the FNO1d Model](#training-the-fno1d-model)
- [Testing FNO1d Model Results](#testing-fno1d-model-results)
- [Fourier Layers](#fourier-layers)
- [FNO2_1d Model](#fno2_1d-model)
- [Training the FNO2_1d Model](#training-the-fno2_1d-model)
- [Testing FNO2_1d Model Results](#testing-fno2_1d-model-results)
- [Convolutional Neural Networks (CNN)](#convolutional-neural-networks-cnn)
  - [CNN Model](#cnn-model)
  - [Training the CNN Model](#training-the-cnn-model)
  - [Testing CNN Model Results](#testing-cnn-model-results)
- [FNO vs CNN](#fno-vs-cnn)
  - [Comparing Metrics](#comparing-metrics)




## Neural Operators for Solving PDEs

Many scientific and engineering problems require solving systems of partial differential equations (PDEs). However, traditional PDE solvers, such as finite element methods (FEM) and finite difference methods (FDM), rely on discretizing the computational domain into a fine mesh, which can be computationally expensive and inefficient.

**Neural operators** offer an alternative approach by leveraging neural networks to learn solution operators for PDEs. Instead of solving PDEs iteratively on a mesh, neural operators take initial or boundary conditions as input and directly produce the solution, akin to an image-to-image transformation.

Unlike standard methods, neural operators are mesh-independent. They can be trained on one discretization and evaluated on another, as they operate in function space rather than learning discrete vectors.

| **Conventional PDE Solvers** | **Neural Operators** |
|-----------------------------|----------------------|
| Solve a single instance at a time | Learn a family of PDEs |
| Require explicit equation formulation | Black-box, data-driven approach |
| Speed-accuracy trade-off based on resolution | Resolution- and mesh-invariant |
| Computationally expensive on fine grids, faster on coarse grids | Slow to train but fast to evaluate |


## Fourier Neural Operators (FNOs)

**Fourier Neural Operators (FNOs)** are a deep learning approach for solving PDEs efficiently by learning function-to-function mappings. Unlike traditional numerical solvers, FNOs operate in the Fourier domain, making them significantly faster and more scalable for high-dimensional PDEs.

Fourier Neural Operators (FNOs) rely on **operator learning**, where we attempt to learn mappings between functions rather than just input-output pairs. It generalizes neural networks to work with infinite-dimensional spaces, such as solutions to differential equations. Particularly, it learns a transformation $G$ that maps one function to another:

$$G: u(x) \to v(x)$$  

where $u(x)$ is the input function and $v(x)$ is the output function.

<div align="center">
  <img src="./figures/FNO_1.png" alt="Sample of a PDE">
</div>

PDE solutions function as operators that map between function spaces, taking inputs like initial conditions, boundary conditions, and source terms to produce the corresponding solution. 

### Fourier Layer
Since the inputs and outputs of partial differential equations (PDEs) are continuous functions, representing them in Fourier space is often more efficient.

In the spatial domain, convolution corresponds to pointwise multiplication in the Fourier domain. To apply the (global) convolution operator, we first perform a Fourier transform, followed by a linear transformation, and then an inverse Fourier transform.

<div align="center">
  <img src="./figures/fourier_layer.png" alt="Fourier Layers">
</div>

The Fourier layer just consists of three steps:

1. Fourier transform (using FFT) $F$

2. Linear transform on the lower Fourier modes $R$

3. Inverse Fourier transform $F^{-1}$


We incorporate a Fourier Layer in our implementation `SpectralConv1d`designed to perform convolution in the Fourier domain rather than in the spatial domain.





## Data Exploration and Visualization

Our data consists of the initial conditions and mapped solutions. It has the following dimensions:

```python
Input Data Shape: torch.Size([1000, 1001, 2])
Output Data Shape: torch.Size([1000, 1001])
Number of Samples: 1000
Time Steps: 1001
Number of Features in Input: 2
```

<div align="center">
  <img src="./figures/sample_629.png" alt="Sample 629">
</div>


## Deep Learning Models

### 1. FNO1d Model

<div align="center">
  <img src="./models_viz/figs/FNO_Model.png" alt="FNO_Model.png">
</div>

# Training the FNO1d Model

We train the model using Adam optimizer and StepLR scheduler for learning rate adjustment. We used the following evaluation metrics:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score (Coefficient of Determination)
  - Relative L2 Error (%)

```python
learning_rate = 0.001
epochs = 30
step_size = 50
gamma = 0.5
```

```python
modes = 16
width = 64
```
The model is initialized and then trained using the specified hyperparameters. 
```python
fno = FNO1d(modes, width)
```

```python
optimizer = Adam(fno.parameters(), lr=learning_rate, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Loss function
mse_loss_fn = torch.nn.MSELoss()
mae_loss_fn = torch.nn.L1Loss()

train_mse_history = []
train_mae_history = []
train_rmse_history = []

test_mse_history = []
test_mae_history = []
test_r2_history = []
test_rmse_history = []
test_relative_l2_history = []

freq_print = 1

# Training Loop
for epoch in range(epochs):
    fno.train()  
    train_mse, train_mae, train_rmse = 0.0, 0.0, 0.0 

    # Loop over batches in the training set
    for step, (input_batch, output_batch) in enumerate(training_set):
        optimizer.zero_grad()  # Reset gradients

        # Get predictions
        output_pred_batch = fno(input_batch).squeeze(2)

        loss_mse = mse_loss_fn(output_pred_batch, output_batch)
        loss_mse.backward()  # Backpropagation

        optimizer.step()  # Update model parameters

        # Compute additional training metrics
        loss_mae = mae_loss_fn(output_pred_batch, output_batch) 
        loss_rmse = torch.sqrt(loss_mse)

        train_mse += loss_mse.item()
        train_mae += loss_mae.item()
        train_rmse += loss_rmse.item()

    train_mse /= len(training_set)
    train_mae /= len(training_set)
    train_rmse /= len(training_set)

    # Store training losses
    train_mse_history.append(train_mse)
    train_mae_history.append(train_mae)
    train_rmse_history.append(train_rmse)

    # Update learning rate
    scheduler.step()

    # Evaluation on the test set
    fno.eval() 
    test_mse, test_mae, test_rmse, test_r2, test_relative_l2 = 0.0, 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for step, (input_batch, output_batch) in enumerate(testing_set):
            output_pred_batch = fno(input_batch).squeeze(2)

            mse = mse_loss_fn(output_pred_batch, output_batch)
            mae = mae_loss_fn(output_pred_batch, output_batch)
            rmse = torch.sqrt(mse)

            relative_l2 = (torch.mean((output_pred_batch - output_batch) ** 2) / torch.mean(output_batch ** 2)) ** 0.5 * 100

            ss_total = torch.sum((output_batch - torch.mean(output_batch)) ** 2)
            ss_residual = torch.sum((output_batch - output_pred_batch) ** 2)
            r2_score = 1 - (ss_residual / ss_total)

            # Accumulate test losses
            test_mse += mse.item()
            test_mae += mae.item()
            test_rmse += rmse.item()
            test_relative_l2 += relative_l2.item()
            test_r2 += r2_score.item()

        test_mse /= len(testing_set)
        test_mae /= len(testing_set)
        test_rmse /= len(testing_set)
        test_relative_l2 /= len(testing_set)
        test_r2 /= len(testing_set)

        # Store test losses
        test_mse_history.append(test_mse)
        test_mae_history.append(test_mae)
        test_rmse_history.append(test_rmse)
        test_relative_l2_history.append(test_relative_l2)
        test_r2_history.append(test_r2)

    # Print training and test metrics in a table format
        if epoch % freq_print == 0:
          print(f"\nEpoch: {epoch}")
          print(f"{'Metric':<20}{'Train':<20}{'Test'}")
          print("-" * 60)
          print(f"{'MSE':<20}{train_mse:<20f}{test_mse:.6f}")
          print(f"{'MAE':<20}{train_mae:<20f}{test_mae:.6f}")
          print(f"{'RMSE':<20}{train_rmse:<20f}{test_rmse:.6f}")
          print(f"{'R²':<20}{'-':<20}{test_r2:.6f}")  # R² is only for the test set
          print(f"{'Relative L2 (%)':<20}{'-':<20}{test_relative_l2:.6f}%")  # Only for test
```
Last Iterated Outputs:

```
.
.
.
Epoch: 28
Metric              Train               Test
------------------------------------------------------------
MSE                 0.000729            0.002130
MAE                 0.014293            0.019510
RMSE                0.025856            0.042593
R²                  -                   0.996894
Relative L2 (%)     -                   5.143578%

Epoch: 29
Metric              Train               Test
------------------------------------------------------------
MSE                 0.000624            0.002351
MAE                 0.013763            0.019859
RMSE                0.024232            0.043836
R²                  -                   0.996572
Relative L2 (%)     -                   5.293022%
```

To further understand the model architecture, we summarize its structure using the function below. This summary provides insights into the number of layers, parameters, and computational complexity, helping evaluate the impact of architectural modifications on model performance.

```python
def model_summary(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    total_params+=params
    print(table)
    print(f'Total Trainable Params: {total_params}')
    return total_params
```

```python
model_summary(fno)
```

```
+---------------------+------------+
|       Modules       | Parameters |
+---------------------+------------+
|   linear_p.weight   |    128     |
|    linear_p.bias    |     64     |
|   spect1.weights1   |   65536    |
|   spect2.weights1   |   65536    |
|   spect3.weights1   |   65536    |
|     lin0.weight     |    4096    |
|      lin0.bias      |     64     |
|     lin1.weight     |    4096    |
|      lin1.bias      |     64     |
|     lin2.weight     |    4096    |
|      lin2.bias      |     64     |
|   linear_q.weight   |    2048    |
|    linear_q.bias    |     32     |
| output_layer.weight |     32     |
|  output_layer.bias  |     1      |
+---------------------+------------+
Total Trainable Params: 211393
211393
```

# Testing FNO1d Model Results

In the following section, we explore the visual results of the Fourier Neural Operator (FNO) model on the test dataset. The predicted outputs are compared with the ground truth, providing insights into the model's ability to learn and generalize patterns in 1D regression tasks.  

These visualizations highlight the model's performance by showcasing the predicted and actual values, helping assess accuracy and error distribution across different test samples.  

The following plots display the model's predictions:

<div align="center">
  <img src="./figures/true_vs_approx.png" alt="FNO1d Predictions">
</div>


# Fourier Layers

To evaluate how the **Fourier Neural Operator's (FNO)** predictive capability is affected, we developed a new model, **FNO2_1d**, using the same hyperparameters but with additional **Fourier layers**. By increasing the model depth, we aim to assess whether the extra layers enhance accuracy and generalization in 1D regression tasks.  

The following sections compare the performance of **FNO1d** and **FNO2_1d**, analyzing their predictions and error metrics on the test dataset.

```python
fno_2 = FNO2_1d(modes, width)
```

```python
model_summary(fno_2)
```

```
+---------------------+------------+
|       Modules       | Parameters |
+---------------------+------------+
|   linear_p.weight   |    128     |
|    linear_p.bias    |     64     |
|   spect1.weights1   |   65536    |
|   spect2.weights1   |   65536    |
|   spect3.weights1   |   65536    |
|   spect4.weights1   |   65536    |
|   spect5.weights1   |   65536    |
|     lin0.weight     |    4096    |
|      lin0.bias      |     64     |
|     lin1.weight     |    4096    |
|      lin1.bias      |     64     |
|     lin2.weight     |    4096    |
|      lin2.bias      |     64     |
|     lin3.weight     |    4096    |
|      lin3.bias      |     64     |
|     lin4.weight     |    4096    |
|      lin4.bias      |     64     |
|   linear_q.weight   |    2048    |
|    linear_q.bias    |     32     |
| output_layer.weight |     32     |
|  output_layer.bias  |     1      |
+---------------------+------------+
Total Trainable Params: 350785
350785
```
FNO2_1d Model Predictions:

<div align="center">
  <img src="./figures/true_vs_approx_fno2.png" alt="FNO2d Predictions">
</div>

As we can see, increasing the number of **Fourier layers** did not significantly improve the model's predictive ability. While the additional layers slightly influenced the results, the overall effect on accuracy and generalization was minimal. This suggests that simply increasing the depth of the Fourier layers may not always lead to better performance and that other factors, such as data quality, regularization, or alternative architectural modifications, might play a more crucial role in enhancing the model’s predictions.


# Convolutional Neural Networks (CNN)

In this section, we investigate the use of Convolutional Neural Networks (CNNs) for solving the Allen-Cahn equation.

The goal is to evaluate how two model architectures (FNOs & CNNs) perform when applied to the same problem, focusing on their ability to approximate solutions to the Allen-Cahn equation effectively.

## CNN Model

```python
class CNNModel(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(CNNModel, self).__init__()

        # Expanded number of layers and increased the number of filters
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1) 
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1) 
        self.conv6 = nn.Conv1d(256, 512, kernel_size=3, padding=1)  

        self.conv7 = nn.Conv1d(512, output_channels, kernel_size=3, padding=1)  # Output layer

        # Adaptive average pooling to adjust the output size to 1001
        self.pool = nn.AdaptiveAvgPool1d(1001)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Ensure x is 3D: [batch_size, channels, sequence_length]
        if len(x.shape) == 4:  # Check for extra dimension
            x = x.squeeze(-1)  # Remove the extra dimension

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.conv7(x)  # Final convolution layer

        x = self.pool(x)  # Resize to 1001 length if necessary

        return x
```

## Training the CNN Model

```python
train_mse_history = []
train_mae_history = []
train_rmse_history = []
train_relative_l2_history = []

test_mse_history = []
test_mae_history = []
test_rmse_history = []
test_relative_l2_history = []  
test_r2_history = []
```

```python
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_model = CNNModel(input_channels=2, output_channels=1).to(device)

# Define optimizer and scheduler
optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Loss functions
mse_loss_fn = nn.MSELoss()
mae_loss_fn = nn.L1Loss()

# Training Loop
for epoch in range(epochs):
    cnn_model.train()
    train_mse, train_mae, train_rmse, train_relative_l2 = 0.0, 0.0, 0.0, 0.0

    for step, (input_batch, output_batch) in enumerate(training_set):
        optimizer.zero_grad()

        input_batch = input_batch.view(input_batch.shape[0], 2, -1)  # Ensure correct shape
        output_pred_batch = cnn_model(input_batch).squeeze(1)

        # Compute loss values
        loss_mse = mse_loss_fn(output_pred_batch, output_batch)
        loss_mse.backward()
        optimizer.step()

        loss_mae = mae_loss_fn(output_pred_batch, output_batch)
        loss_rmse = torch.sqrt(loss_mse)

        relative_l2 = torch.norm(output_pred_batch - output_batch) / torch.norm(output_batch)

        train_mse += loss_mse.item()
        train_mae += loss_mae.item()
        train_rmse += loss_rmse.item()
        train_relative_l2 += relative_l2.item()

    train_mse /= len(training_set)
    train_mae /= len(training_set)
    train_rmse /= len(training_set)
    train_relative_l2 /= len(training_set)

    train_mse_history.append(train_mse)
    train_mae_history.append(train_mae)
    train_rmse_history.append(train_rmse)
    train_relative_l2_history.append(train_relative_l2)

    scheduler.step()

    # Evaluation
    cnn_model.eval()
    test_mse, test_mae, test_rmse, test_r2, test_relative_l2 = 0.0, 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for step, (input_batch, output_batch) in enumerate(testing_set):
            input_batch = input_batch.view(input_batch.shape[0], 2, -1)  
            output_pred_batch = cnn_model(input_batch).squeeze(1)

            mse = mse_loss_fn(output_pred_batch, output_batch)
            mae = mae_loss_fn(output_pred_batch, output_batch)
            rmse = torch.sqrt(mse)

            # Calculate Relative L2 Error
            relative_l2 = torch.norm(output_pred_batch - output_batch) / torch.norm(output_batch)

            ss_total = torch.sum((output_batch - torch.mean(output_batch)) ** 2)
            ss_residual = torch.sum((output_batch - output_pred_batch) ** 2)
            r2_score = 1 - (ss_residual / ss_total)

            test_mse += mse.item()
            test_mae += mae.item()
            test_rmse += rmse.item()
            test_relative_l2 += relative_l2.item()
            test_r2 += r2_score.item()

        # Average out the metrics
        test_mse /= len(testing_set)
        test_mae /= len(testing_set)
        test_rmse /= len(testing_set)
        test_relative_l2 /= len(testing_set)
        test_r2 /= len(testing_set)

        test_mse_history.append(test_mse)
        test_mae_history.append(test_mae)
        test_rmse_history.append(test_rmse)
        test_relative_l2_history.append(test_relative_l2)
        test_r2_history.append(test_r2)

   # Print training and test metrics in a table format
        if epoch % freq_print == 0:
          print(f"\nEpoch: {epoch}")
          print(f"{'Metric':<20}{'Train':<20}{'Test'}")
          print("-" * 60)
          print(f"{'MSE':<20}{train_mse:<20f}{test_mse:.6f}")
          print(f"{'MAE':<20}{train_mae:<20f}{test_mae:.6f}")
          print(f"{'RMSE':<20}{train_rmse:<20f}{test_rmse:.6f}")
          print(f"{'R²':<20}{'-':<20}{test_r2:.6f}")  # R² is only for the test set
          print(f"{'Relative L2 (%)':<20}{'-':<20}{test_relative_l2:.6f}%")  # Only for test
```

Last Iterated Outputs:

```
.
.
.
Epoch: 28
Metric              Train               Test
------------------------------------------------------------
MSE                 0.517000            0.606153
MAE                 0.606281            0.644920
RMSE                0.718095            0.776237
R²                  -                   0.115099
Relative L2 (%)     -                   0.937226%

Epoch: 29
Metric              Train               Test
------------------------------------------------------------
MSE                 0.501254            0.584649
MAE                 0.597721            0.657924
RMSE                0.707504            0.762959
R²                  -                   0.146385
Relative L2 (%)     -                   0.921221%
```

```python
model_summary(cnn_model)
```

```
+--------------+------------+
|   Modules    | Parameters |
+--------------+------------+
| conv1.weight |    192     |
|  conv1.bias  |     32     |
| conv2.weight |    6144    |
|  conv2.bias  |     64     |
| conv3.weight |   24576    |
|  conv3.bias  |    128     |
| conv4.weight |   49152    |
|  conv4.bias  |    128     |
| conv5.weight |   98304    |
|  conv5.bias  |    256     |
| conv6.weight |   393216   |
|  conv6.bias  |    512     |
| conv7.weight |    1536    |
|  conv7.bias  |     1      |
+--------------+------------+
Total Trainable Params: 574241
574241
```

CNN Model Predictions:

<div align="center">
  <img src="./figures/true_vs_approx_cnn.png" alt="CNN Predictions">
</div>


# FNO vs CNN

In this section, we compare the performance of Fourier Neural Operators (FNO) and Convolutional Neural Networks (CNNs) in solving the Allen-Cahn equation.

## Comparing Metrics

The Testing Loss Curve compares the performance of the FNO, FNO_2, and CNN models over 30 epochs.

<div align="center">
  <img src="./figures/testing_loss_curve_comparison.png" alt="CNN Predictions">
</div>


Below is a function that creates a summary table comparing multiple models based on their evaluation metrics. 

| Model Index | Model     | Test MSE  | Test MAE  | Test RMSE | Test R²   | Test Relative L2 (%) | Min Test Loss | Total Parameters |
|------------|----------|-----------|-----------|-----------|-----------|----------------------|---------------|------------------|
| 0          | FNO1d    | $$0.002351$$  | $$0.019859$$  | $$0.043836$$  | $$0.996572$$  | $$5.293022$$             | $$0.002130$$      | $$211393$$           |
| 1          | FNO2_1d  | $$0.003570$$  | $$0.029459$$  | $$0.057029$$  | $$0.994789$$  | $$6.972500$$             | $$0.001887$$      | $$350785$$           |
| 2          | CNNModel | $$0.584649$$  | $$0.657924$$  | $$0.762959$$  | $$0.146385$$  | $$0.921221$$             | $$0.567342$$      | $$574241$$           |


As we can see, FNO1d outperforms both FNO2_1d and CNNModel across all metrics. It achieves the lowest error, highest R² score, and best overall predictive performance. Moreover, adding more Fourier layers in FNO2_1d did not lead to improvements, as its error metrics are slightly worse than FNO1d. The CNN model performs significantly worse than both FNO models, despite having the highest number of parameters.


