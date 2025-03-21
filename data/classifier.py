from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def plot():
    df = pd.read_csv('UP_CLASSIFIER.csv')

    # Convert start_time to datetime
    df['start_time'] = pd.to_datetime(df['start_time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

    # Convert datetime to seconds (elapsed time since the first timestamp)
    df['start_time'] = (df['start_time'] - df['start_time'].min()).dt.total_seconds()

    # Convert other columns to numeric
    df['accel_magnitude_sma'] = pd.to_numeric(df['accel_magnitude_sma'], errors='coerce')
    df['gyro_magnitude_sma'] = pd.to_numeric(df['gyro_magnitude_sma'], errors='coerce')

    # Drop NaN values
    df.dropna(subset=['start_time', 'accel_magnitude_sma', 'gyro_magnitude_sma'], inplace=True)

    # Ensure filename is a string
    df['filename'] = df['filename'].astype(str)

    # Group by filename
    grouped = df.groupby('filename')

    # Read existing filenames from UP_fall_timestamps.csv to know what has been processed
    filenames_df = pd.read_csv('UP_fall_timestamps.csv') 
    filenames_df['filename'] = filenames_df['filename'].astype(str)  # Ensure consistency
    filenames = set(filenames_df['filename'])  # Convert to set for fast lookup

    # Create a new CSV file to store the fall events
    new_csv_file = 'UP_fall_timestamps_new.csv'

    # Plot the accel_magnitude_sma and gyro_magnitude_sma for each filename
    for filename, group in grouped:
        if filename not in filenames:
            continue

        # Sort by start_time
        group = group.sort_values(by='start_time')

        # Extract values for plotting
        start_times = group['start_time'].values
        accel_magnitudes = group['accel_magnitude_sma'].values
        gyro_magnitudes = group['gyro_magnitude_sma'].values

        # Debug: Print data being plotted
        print(f"Filename: {filename}")
        print("start_time:", start_times)
        print("accel_magnitude_sma:", accel_magnitudes)
        print("gyro_magnitude_sma:", gyro_magnitudes)

        # Scale the data per group
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler2 = MinMaxScaler(feature_range=(0, 1))
        group[['accel_magnitude_sma']] = scaler.fit_transform(group[['accel_magnitude_sma']])
        group[['gyro_magnitude_sma']] = scaler2.fit_transform(group[['gyro_magnitude_sma']])

        # Plot
        plt.figure()
        plt.plot(start_times, accel_magnitudes, label='accel_magnitude_sma')
        plt.plot(start_times, gyro_magnitudes, label='gyro_magnitude_sma')
        plt.legend()
        plt.gcf().set_size_inches(8, 8)
        plt.grid()
        plt.title(filename)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Magnitude')
        plt.show()
        plt.close()  # Close the figure to free up memory

        # Take start_time and end_time from user input and overwrite the values in UP_fall_timestamps.csv
        start, end = input(f"Enter start_time and end_time for {filename} (separated by a space): ").split()
        start, end = float(start), float(end)

        # Create a DataFrame with the new fall event
        fall_event = pd.DataFrame({'filename': [filename], 'start_time': [start], 'end_time': [end]})

        # Append the fall event to the new CSV file (create the file with header if it's empty)
        fall_event.to_csv(new_csv_file, mode='a', header=not pd.io.common.file_exists(new_csv_file), index=False)

plot()
