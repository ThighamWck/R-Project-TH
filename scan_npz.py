import numpy as np

# Load the original file
npz_path = "300625_train_data_32_transform_2500_1.npz"
data = np.load(npz_path)
data_array = data['data']
columns = data['columns']

# Map column names to indices
col_idx = {name: i for i, name in enumerate(columns)}

# Clip rho values to [0, 1]
rho = data_array[:, col_idx['rho_val']]
rho_clipped = np.clip(rho, 0, 1)
data_array[:, col_idx['rho_val']] = rho_clipped

# Save to a new file
new_npz_path = "300625_train_data_32_transform_2500_1_clipped.npz"
np.savez(new_npz_path, data=data_array, columns=columns)

print(f"Clipped rho values to [0, 1] and saved to {new_npz_path}")

# Extract relevant columns
rho = data_array[:, col_idx['rho_val']]
b1 = data_array[:, col_idx['b1']]
b2 = data_array[:, col_idx['b2']]

# Find out-of-bounds values
rho_out_of_bounds_mask = (rho < 0) | (rho > 1)
b1_large_mask = np.abs(b1) > 100
b2_large_mask = np.abs(b2) > 100

rho_out_of_bounds = np.sum(rho_out_of_bounds_mask)
b1_large = np.sum(b1_large_mask)
b2_large = np.sum(b2_large_mask)

print(f"Total samples: {len(rho)}")
print(f"rho out of [0, 1]: {rho_out_of_bounds}")
print(f"b1 > 100 or < -100: {b1_large}")
print(f"b2 > 100 or < -100: {b2_large}")

print(f"\nrho min: {rho.min()}, max: {rho.max()}")
print(f"b1 min: {b1.min()}, max: {b1.max()}")
print(f"b2 min: {b2.min()}, max: {b2.max()}")

# Print actual extreme values (up to 10 for each)
if rho_out_of_bounds > 0:
    print("\nExample rho out of bounds:", rho[rho_out_of_bounds_mask][:10])
    print("All extreme rho values:", rho[rho_out_of_bounds_mask])
if b1_large > 0:
    print("\nExample b1 large:", b1[b1_large_mask][:10])
    print("All extreme b1 values:", b1[b1_large_mask])
if b2_large > 0:
    print("\nExample b2 large:", b2[b2_large_mask][:10])
    print("All extreme b2 values:", b2[b2_large_mask])