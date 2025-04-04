import numpy as np

# Load Colvars trajectory
data = np.loadtxt("colvar.traj")

# Extract columns: Time (col 0), RMSD (col 1), Gyration (col 2), Bias Potential (last column)
time_steps = data[:, 0].astype(int)  # Convert the first column to integers
filtered_data = np.column_stack((time_steps, data[:, [1, 2, -1]]))  # Stack integer time with other columns

# Save training data in original file
np.savetxt("update_bias.dat", filtered_data, fmt="%d %.6f %.6f %.6f")
print("Training data saved as update_bias.dat")

# Define skipping interval
N = 5  # Change this value to adjust skipping (e.g., N=5 saves every 5th row)

# Select every Nth row
skipped_data = filtered_data[::N]

# Save skipped data to a new file
np.savetxt("skipped_bias.dat", skipped_data, fmt="%d %.6f %.6f %.6f")
print("Skipped data saved as skipped_bias.dat")
