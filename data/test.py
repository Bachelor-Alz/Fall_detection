import os 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import ast

# Load file
preprocessed_data = os.path.join(os.getcwd(), 'preprocessed', 'preprocessed_w50_o40.csv')
df = pd.read_csv(preprocessed_data)

# Ensure correct column names
expected_cols = ['accel_x_list', 'accel_y_list', 'accel_z_list']
for col in expected_cols:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in dataset!")

# Convert string lists to actual lists if needed
for col in expected_cols:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Split data into UMA and non-UMA
uma_df = df[df['filename'].str.contains('UMA', na=False)]
non_uma_df = df[~df['filename'].str.contains('UMA', na=False)]

# Downsample after filtering
uma_df = uma_df.iloc[::10]
non_uma_df = non_uma_df.iloc[::10]

# Standardize data
scaler_uma = StandardScaler()
scaler_non_uma = StandardScaler()

uma_df[['accel_x_list', 'accel_y_list', 'accel_z_list']] = scaler_uma.fit_transform(uma_df[['accel_x_list', 'accel_y_list', 'accel_z_list']])
non_uma_df[['accel_x_list', 'accel_y_list', 'accel_z_list']] = scaler_non_uma.fit_transform(non_uma_df[['accel_x_list', 'accel_y_list', 'accel_z_list']])

# Plot values for accel xyz between datasets
fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

for i, axis in enumerate(['x', 'y', 'z']):
    axes[i].plot(uma_df[f'accel_{axis}_list'], label=f'UMA accel_{axis}', alpha=0.5)  # Set alpha for transparency
    axes[i].plot(non_uma_df[f'accel_{axis}_list'], label=f'Non-UMA accel_{axis}', alpha=0.5)  # Set alpha for transparency
    axes[i].legend()
    axes[i].set_title(f'Accel {axis.upper()} Comparison')
    axes[i].set_ylabel('Acceleration')

# Adjust plot size and layout
plt.gcf().set_size_inches(12, 12)
plt.xlabel('Time')
plt.tight_layout()
plt.show()
