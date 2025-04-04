import os
import numpy as np
import torch
import gpytorch

# File paths
UPDATED_BIAS_FILE = "update_bias.dat"  # Bias history file (updated)
MODEL_FILE = "gpr_model_gpu.pth"  # GPU-trained GPR model

# Simulation parameters
step_size = 1000  # Bias update interval
update_interval = 50000  # Steps per update cycle
total_steps = 50000000  # Total simulation steps

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> Using device: {device}")

# Step 1: Get last recorded step from update_bias.dat
def get_last_step():
    if os.path.exists(UPDATED_BIAS_FILE):
        bias_data = np.loadtxt(UPDATED_BIAS_FILE)
        if bias_data.size == 0:
            return 5000000  # Default start step if file is empty
        return int(bias_data[-1, 0])  # Get last recorded step
    return 5000000  # Start at 5M steps if file doesn't exist

last_step = get_last_step()
start_step = last_step + step_size
end_step = min(start_step + update_interval, total_steps)

if start_step > total_steps:
    print(">>> Simulation complete. No further bias updates needed.")
    exit()

# Step 2: Load Trained GPR Model
if not os.path.exists(MODEL_FILE):
    print(">>> ERROR: Trained GPR model not found! Exiting...")
    exit()

checkpoint = torch.load(MODEL_FILE, map_location=device)
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.cov_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.cov_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Load model structure
model = GPModel(torch.zeros(1, 2), torch.zeros(1), likelihood).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
model.eval()
likelihood.eval()
print(">>> Loaded GPR model on GPU for bias prediction.")

# Step 3: Generate New Time Steps
new_time_steps = np.arange(start_step, end_step + step_size, step_size)

# Load existing CVs from update_bias.dat or generate new ones
if os.path.exists(UPDATED_BIAS_FILE):
    bias_data = np.loadtxt(UPDATED_BIAS_FILE)
    cv1 = bias_data[:, 1]
    cv2 = bias_data[:, 2]
else:
    cv1, cv2 = np.random.uniform(0, 1, len(new_time_steps)), np.random.uniform(0, 1, len(new_time_steps))

cv1_new = np.random.uniform(np.min(cv1), np.max(cv1), len(new_time_steps))
cv2_new = np.random.uniform(np.min(cv2), np.max(cv2), len(new_time_steps))

# Convert data to GPU tensor
X_new = torch.tensor(np.column_stack((cv1_new, cv2_new)), dtype=torch.float32).to(device)

# Step 4: Predict Bias using GPR
with torch.no_grad():
    pred = likelihood(model(X_new))
    bias_pred = pred.mean.cpu().numpy()

# Add small noise to avoid overfitting
bias_pred += np.random.normal(0, 0.005, size=bias_pred.shape)

# Step 5: Save Updated Bias History
new_data = np.column_stack((new_time_steps, cv1_new, cv2_new, bias_pred))

# Append to update_bias.dat
if os.path.exists(UPDATED_BIAS_FILE):
    existing_data = np.loadtxt(UPDATED_BIAS_FILE)
    combined_data = np.vstack((existing_data, new_data))
else:
    combined_data = new_data

np.savetxt(UPDATED_BIAS_FILE, combined_data, fmt="%d %.6f %.6f %.6f")

print(f">>> Successfully updated bias in {UPDATED_BIAS_FILE} for steps {start_step} to {end_step}")
