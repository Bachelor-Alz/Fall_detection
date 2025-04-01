
import os
import re
import FallDetector
from data.LoadData import UmaFallLoader, UpFallLoader, WedaFallLoader
import pandas as pd

def fix_multiple_periods(value):
    """Fix values with multiple periods (e.g., '31.203.125' -> '31.203125')"""
    if isinstance(value, str):
        value = re.sub(r'(\d+)\.(\d+)\.(\d+)', r'\1\.\2\3', value)
    return value

columns_to_use = ['gyro_x_list', 'gyro_y_list', 'gyro_z_list', 'accel_x_list', 'accel_y_list', 'accel_z_list']

uma_loader = UmaFallLoader(os.path.join(os.getcwd(), 'UMAFall'), 'UMA_fall_timestamps.csv')
weda_loader = WedaFallLoader(os.path.join(os.getcwd(), 'WEDAFall'), 'WEDA_fall_timestamps.csv')
up_fall_loader = UpFallLoader(os.path.join(os.getcwd(), 'UpFall'), 'UP_fall_timestamps.csv')
fall_detector = FallDetector.FallDetector(window_size=80, overlap=75, data_loaders=[uma_loader, weda_loader, up_fall_loader])
df = fall_detector.load_data()

for col in columns_to_use:
    df[col] = df[col].apply(fix_multiple_periods)
    df[col] = pd.to_numeric(df[col], errors='coerce')

uma_df = df[df['filename'].str.contains('UMA', na=False)]
up_df = df[~df['filename'].str.contains('UMA|U', na=False)]
weda_df = df[df['filename'].str.contains(r'U\d{2}_R\d{2}', na=False, regex=True)]

# Find min, max and mean


def find_min_max_mean(df, columns):
    min_max_mean = {}
    for col in columns:
        min_max_mean[col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean()
        }
    return min_max_mean

uma_min_max_mean = find_min_max_mean(uma_df, columns_to_use)
up_min_max_mean = find_min_max_mean(up_df, columns_to_use)
weda_min_max_mean = find_min_max_mean(weda_df, columns_to_use)

def pretty_print_min_max_mean(min_max_mean):
    for col, values in min_max_mean.items():
        print(f"{col}: Min = {values['min']}, Max = {values['max']}, Mean = {values['mean']}")

print("UMA Data Min, Max, Mean:")
pretty_print_min_max_mean(uma_min_max_mean)
print("\nUP Data Min, Max, Mean:")
pretty_print_min_max_mean(up_min_max_mean)
print("\nWEDA Data Min, Max, Mean:")
pretty_print_min_max_mean(weda_min_max_mean)