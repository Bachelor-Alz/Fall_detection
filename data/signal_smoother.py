import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Load the data, dropping unnecessary columns
df = pd.read_csv('preprocessed_data.csv', usecols=lambda col: col not in ['start_time', 'end_time'], low_memory=False)

# Filter based on the filename conditions
uma_df = df[df['filename'].str.contains('UMA', na=False)]
up_df = df[~df['filename'].str.contains('UMA|U', na=False)]
weda_df = df[df['filename'].str.contains(r'U\d{2}_R\d{2}', na=False, regex=True)]

# List of columns to process (accelerometer and gyroscope data)
columns = ['accel_x_list', 'accel_y_list', 'accel_z_list', 'gyro_x_list', 'gyro_y_list', 'gyro_z_list']

# Function to apply Gaussian smoothing to the raw data
def apply_gaussian_filter(data, sigma=3):
    return gaussian_filter1d(data, sigma=sigma)


# Create a function to plot raw vs. smoothed data for each unique filename
def plot_smoothing_comparison(df, columns, sigma=3):
    counter = 0
    for filename in df['filename'].unique():
        if counter == 100 : break
        counter += 1
        # Filter data by filename
        file_data = df[df['filename'] == filename]
        
        # Plot for each column (accelerometer and gyroscope)
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(columns):
            # Extract the column data
            raw_data = file_data[col].values  # Get the numeric values directly

            # Apply Gaussian smoothing
            smoothed_data = apply_gaussian_filter(raw_data, sigma=sigma)

            # Plot raw and smoothed data
            plt.subplot(len(columns), 1, i+1)
            plt.plot(raw_data, label='Raw Data', alpha=0.6)
            plt.plot(smoothed_data, label=f'Smoothed Data (sigma={sigma})' , linewidth=2)
            plt.title(f'{filename} - {col}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
            plt.grid(True)

        plt.tight_layout()
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{filename}.png')  # Display the plot
        plt.close()  # Close the figure to free up memory

# Plot raw vs. smoothed data for UMA dataset (you can change this to `up_df` or `weda_df` as needed)
plot_smoothing_comparison(up_df, columns)