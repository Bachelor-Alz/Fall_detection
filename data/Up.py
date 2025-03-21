import pandas as pd
import os

# Load dataset, skipping the first row
full_up_pd = pd.read_csv('CompleteDataSet.csv', skiprows=1)

# Extract relevant columns
time_col = full_up_pd.iloc[:, [0]]  # First column as DataFrame
sensor_data = full_up_pd.iloc[:, 29:35]  # Accel and gyro columns

# Combine into a new DataFrame
df = pd.concat([time_col, sensor_data], axis=1)

# Rename columns
df.columns = ['time', 'accel_x_list', 'accel_y_list', 'accel_z_list', 'gyro_x_list', 'gyro_y_list', 'gyro_z_list']

# Extract grouping columns
df[['subject', 'activity', 'trial']] = full_up_pd.iloc[:, -4:-1]

# for every row assign the filename as {activity}_{subject}_{trail}
df['filename'] = df.apply(lambda row: f"{row['subject']}_{row['activity']}_{row['trial']}.csv", axis=1)

# Define output directorie
fall_dir = 'UpFall/Fall'
adl_dir = 'UpFall/ADL'

# Ensure directories exist
os.makedirs(fall_dir, exist_ok=True)
os.makedirs(adl_dir, exist_ok=True)

# Group by activity, subject, and trial and save
columns_to_save = ['time', 'accel_x_list', 'accel_y_list', 'accel_z_list', 'gyro_x_list', 'gyro_y_list', 'gyro_z_list', 'filename']



for (subject, activity, trial), group in df.groupby(['subject', 'activity', 'trial']):
    filename = f"{subject}_{activity}_{trial}.csv"  # Generate filename
    save_path = fall_dir if activity in range(1, 6) else adl_dir
    group[columns_to_save].to_csv(os.path.join(save_path, filename), index=False)  # Narrow columns only here
