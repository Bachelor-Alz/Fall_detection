from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv('features/features_w80_o75.csv')
timestamps = pd.read_csv('UMA_fall_timestamps.csv')

# Ensure that the 'start_time' column in df is of datetime type
df['start_time'] = pd.to_datetime(df['start_time'])

# Get UMA features
uma_features = df[df['filename'].str.contains('UMA')]

for filename, group in uma_features.groupby('filename'):
    # Get the gyro and accel data
    time = group['start_time'].values
    gyro = group['gyro_magnitude_sma'].values
    accel = group['accel_magnitude_sma'].values

    # Convert 'time' to numeric values in milliseconds (milliseconds since Unix epoch)
    time_numeric = (pd.to_datetime(time) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')

    # Get the corresponding timestamps for the current filename
    fall = timestamps[timestamps['filename'] == filename]
    if fall.empty:
        print(f"Warning: No fall data found for {filename}. Skipping...")
        continue

    new_fall_times = pd.read_csv('BLA.csv')
    if filename in new_fall_times['filename'].values:
        continue

    # Ensure the lengths match
    if len(time) != len(gyro) or len(time) != len(accel):
        print(f"Warning: Length mismatch for {filename}. Skipping...")
        continue

    # Get the fall start_time and end_time and convert them to numeric values (milliseconds)
    fall_start = pd.to_datetime(fall.iloc[0]['start_time'], unit='s')
    fall_end = pd.to_datetime(fall.iloc[0]['end_time'], unit='s')

    # Convert fall start and end to numeric values in milliseconds (milliseconds since Unix epoch)
    fall_start_numeric = (fall_start - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')
    fall_end_numeric = (fall_end - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')

    # Plotting
    plt.plot(time_numeric, gyro, label='Gyro')
    plt.plot(time_numeric, accel, label='Accel')
    plt.axvline(fall_start_numeric, color='r', linestyle='--', label='Fall start')
    plt.axvline(fall_end_numeric, color='g', linestyle='--', label='Fall end')

    # Formatting the x-axis to handle milliseconds properly
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: pd.to_datetime(x, unit='ms').strftime('%H:%M:%S.%f')[:-3]))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=False, prune='both'))

    plt.title(filename)
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Magnitude SMA')
    plt.legend()

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.show()

    # Collect user input for new fall event
    start, end = input(f"Enter start_time and end_time for {filename} (separated by a space): ").split()
    start, end = float(start), float(end)

    # Create a DataFrame with the new fall event
    fall_event = pd.DataFrame({'filename': [filename], 'start_time': [start], 'end_time': [end]})

    # Append the fall event to the new CSV file (create the file with header if it's empty)
    fall_event.to_csv('BLA.csv', mode='a', header=not pd.io.common.file_exists('BLA.csv'), index=False)
