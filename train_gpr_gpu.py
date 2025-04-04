import os
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

# File paths
UPDATED_BIAS_FILE = "update_bias.dat"
MODEL_FILE = "gpr_model_gpu.pth"  # Save model in PyTorch format

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> Using device: {device}")

# Step 1: Load Training Data
if not os.path.exists(UPDATED_BIAS_FILE):
    print(">>> ERROR: No data available for training.")
    exit()

bias_data = np.loadtxt(UPDATED_BIAS_FILE)
if bias_data.shape[0] < 10:
    print(">>> ERROR: Not enough data to train GPR model.")
    exit()

# Extract Features (CVs) and Target (Bias Potential)
X_train = torch.tensor(bias_data[:, 1:3], dtype=torch.float32).to(device)  # CV1, CV2
y_train = torch.tensor(bias_data[:, -1], dtype=torch.float32).to(device)  # Bias potential

# Step 2: Define GPR Model using GPyTorch
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.cov_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.cov_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Step 3: Initialize Model and Likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model = GPModel(X_train, y_train, likelihood).to(device)

# Step 4: Train the Model with Convergence Check
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Convergence settings
tolerance = 1e-6  # Convergence tolerance
patience = 30  # Number of iterations to wait before stopping if no improvement
best_loss = float('inf')  # Start with a very high loss
no_improvement_count = 0  # Counter for iterations without improvement

# Loss tracking for plotting
loss_history = []

print(">>> Training GPR model on GPU...")
for i in range(500):  # Training loop
    optimizer.zero_grad()
    output = model(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()

    # Track loss for plotting
    loss_history.append(loss.item())

    # Check if the loss has improved
    if loss.item() < best_loss:
        best_loss = loss.item()
        no_improvement_count = 0  # Reset the counter
    else:
        no_improvement_count += 1

    # Print every 10th iteration
    if i % 10 == 0:
        print(f"Iteration {i}/500 - Loss: {loss.item()}")

    # Early stopping if no improvement for 'patience' iterations
    if no_improvement_count >= patience:
        print(f"Convergence achieved after {i} iterations.")
        break

# Step 5: Save the Trained Model
torch.save({'model_state_dict': model.state_dict(),
            'likelihood_state_dict': likelihood.state_dict()}, MODEL_FILE)

print(f">>> Successfully trained and saved GPU-accelerated GPR model as {MODEL_FILE}")

# Step 6: Save the loss curve plot to a file
plt.plot(range(len(loss_history)), loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')

# Save the plot as a PNG image file
plot_filename = "training_loss_curve.png"
plt.savefig(plot_filename)

# Optionally, you can also save it as a PDF or other formats
# plt.savefig("training_loss_curve.pdf")

print(f">>> Training loss curve has been saved as {plot_filename}")

# Close the plot to free memory
plt.close()
