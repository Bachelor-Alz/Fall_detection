import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Define root directory
current_dir = os.path.dirname(__file__)
datafolder = os.path.join(current_dir, '40Hz')
all_features = []

# Sliding window parameters
WINDOW_SIZE = 2.0  # 2 seconds per window
OVERLAP = 0.5  # 50% overlap

# Function to load and clean data
def load_and_clean_data(accel_file, gyro_file):
    data1 = pd.read_csv(accel_file)
    data2 = pd.read_csv(gyro_file)

    # Rename time columns
    data1 = data1.rename(columns={'accel_time_list': 'time'})
    data2 = data2.rename(columns={'gyro_time_list': 'time'})

    # Merge using an outer join
    merged_data = pd.merge(data1, data2, on='time', how='outer')

    # Sort by time
    merged_data = merged_data.sort_values(by='time')

    # Interpolate missing values
    merged_data = merged_data.interpolate()

    return merged_data

# Function for pre-processing (scaling)
def preprocess_data(df):
    # Initialize the scaler
    scaler = StandardScaler()

    # Apply scaling to the accelerometer and gyroscope data
    accel_columns = ['accel_x_list', 'accel_y_list', 'accel_z_list']
    gyro_columns = ['gyro_x_list', 'gyro_y_list', 'gyro_z_list']
    
    # Fit and transform accelerometer data
    df[accel_columns] = scaler.fit_transform(df[accel_columns])
    
    # Fit and transform gyroscope data
    df[gyro_columns] = scaler.fit_transform(df[gyro_columns])

    return df

# Function to compute features
def extract_features(df, activity_label):
    window_features = []
    
    # Get the min/max time to create windows
    start_time = df['time'].min()
    end_time = df['time'].max()

    # Slide over the data in fixed-size windows
    while start_time + WINDOW_SIZE <= end_time:
        window_df = df[(df['time'] >= start_time) & (df['time'] < start_time + WINDOW_SIZE)]

        if not window_df.empty:
            features = {'start_time': start_time, 'end_time': start_time + WINDOW_SIZE}

            # Compute statistics for each axis
            for axis in ['accel_x_list', 'accel_y_list', 'accel_z_list', 
                         'gyro_x_list', 'gyro_y_list', 'gyro_z_list']:
                if axis in df.columns:
                    features[f'mean_{axis}'] = window_df[axis].mean()
                    features[f'std_{axis}'] = window_df[axis].std()
                    features[f'min_{axis}'] = window_df[axis].min()
                    features[f'max_{axis}'] = window_df[axis].max()
                    features[f'median_{axis}'] = window_df[axis].median()
                    features[f'iqr_{axis}'] = window_df[axis].quantile(0.75) - window_df[axis].quantile(0.25)

                    # Root Mean Square (RMS)
                    features[f'rms_{axis}'] = (window_df[axis] ** 2).mean() ** 0.5

                    # Signal Energy
                    features[f'energy_{axis}'] = (window_df[axis] ** 2).sum()

            # Signal Magnitude Area (SMA)
            features['SMA_accel'] = (window_df[['accel_x_list', 'accel_y_list', 'accel_z_list']].abs().sum(axis=1)).mean()
            features['SMA_gyro'] = (window_df[['gyro_x_list', 'gyro_y_list', 'gyro_z_list']].abs().sum(axis=1)).mean()

            # Assign activity classification
            if('F' in activity_label): 
                features['activity'] = 1
            else: 
                features['activity'] = 0
        
            # Append to feature list
            window_features.append(features)

        # Move to the next window (with overlap)
        start_time += WINDOW_SIZE * (1 - OVERLAP)

    return window_features

# Process each activity folder
for activity_folder in os.listdir(datafolder):
    activity_path = os.path.join(datafolder, activity_folder)
    
    if os.path.isdir(activity_path):
        accel_file, gyro_file = None, None

        for file in os.listdir(activity_path):
            if file.endswith("_accel.csv"):
                accel_file = os.path.join(activity_path, file)
            elif file.endswith("_gyro.csv"):
                gyro_file = os.path.join(activity_path, file)

        if accel_file and gyro_file:
            # Load and clean data
            merged_data = load_and_clean_data(accel_file, gyro_file)

            # Pre-process (scale) data
            preprocessed_data = preprocess_data(merged_data)

            # Extract features with sliding windows
            extracted_windows = extract_features(preprocessed_data, activity_folder)
            all_features.extend(extracted_windows)

# Convert to DataFrame
features_df = pd.DataFrame(all_features)

# Save the extracted features
output_path = os.path.join(current_dir, "feature_extracted_data.csv")
features_df.to_csv(output_path, index=False)

print(f"✅ Feature extraction complete! Saved at {output_path}")
