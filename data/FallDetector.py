import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Union
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message=".*Precision loss occurred.*")


StrPath = Union[str, os.PathLike]

class FallDetector:
    def __init__(self, window_size: int, overlap: int, datafolder: StrPath):
        self.window_size = window_size
        self.overlap = overlap
        self.fall_timestamps = pd.read_csv('fall_timestamps.csv')
        self.datafolder = datafolder
        self.alpha = 0.98

    def load_data(self):
        folders = os.listdir(self.datafolder)
        sorted_folders = sorted(folders)
        df = pd.DataFrame()

        # 50 Mhz
        for folder in sorted_folders:
            files = os.listdir(os.path.join(self.datafolder, folder))
            sorted_files = sorted(files)
            for accel, gyro in zip(sorted_files[0::2], sorted_files[1::2]):
                if "accel" not in accel or "gyro" not in gyro:
                    raise ValueError(f"File mismatch: Expected accel & gyro pair but got '{accel}' and '{gyro}'")

                accel_data = pd.read_csv(os.path.join(self.datafolder, folder, accel))
                gyro_data = pd.read_csv(os.path.join(self.datafolder, folder, gyro))
                
                if accel_data.empty or gyro_data.empty is None:
                    raise ValueError(f"Data is empty")

                accel_data.rename(columns={'accel_time_list':'time'}, inplace=True)
                gyro_data.rename(columns={'gyro_time_list':'time'}, inplace=True)
                merged = pd.merge(how='outer', on='time', left=accel_data, right=gyro_data)
                merged['filename'] = folder + '/' + accel[0:7]
                df = pd.concat([df, merged])


        uma_data = os.path.join(os.getcwd(), 'UMAFall')

        #UMA Fall
        for file in os.listdir(uma_data):
            file_path = os.path.join(uma_data, file)  
            uma_df = pd.read_csv(file_path, skiprows=40, sep=';')  
            uma_df.columns = uma_df.columns.str.strip()
            uma_df.dropna(axis=1, how='all', inplace=True)
            
            NEEDED_SENSOR_ID = 4
            ACCELERO_SENSOR_TYPE = 0
            GYRO_SENSOR_TYPE = 1

            uma_df = uma_df[uma_df['Sensor ID'] == NEEDED_SENSOR_ID]
            uma_df = uma_df[uma_df['Sensor Type'].isin([GYRO_SENSOR_TYPE, ACCELERO_SENSOR_TYPE])]
            uma_df = uma_df.rename(columns={'% TimeStamp': 'time'})
            
            #convert timestamp in ms to seconds
            uma_df['time'] = uma_df['time'] / 1000

            acc_df = uma_df[uma_df['Sensor Type'] == ACCELERO_SENSOR_TYPE]
            gyro_df = uma_df[uma_df['Sensor Type'] == GYRO_SENSOR_TYPE]
            acc_df = acc_df.rename(columns={'X-Axis': 'accel_x_list', 'Y-Axis': 'accel_y_list', 'Z-Axis': 'accel_z_list'})
            gyro_df = gyro_df.rename(columns={'X-Axis': 'gyro_x_list', 'Y-Axis': 'gyro_y_list', 'Z-Axis': 'gyro_z_list'})
            acc_df = acc_df.drop(columns=['Sensor ID', 'Sensor Type', 'Sample No'])
            gyro_df = gyro_df.drop(columns=['Sensor ID', 'Sensor Type', 'Sample No'])
            uma_df = pd.merge(how='outer', on='time', left=acc_df, right=gyro_df)
            uma_df['filename'] = file
            df = pd.concat([df, uma_df])

        return df
    
    def pre_process(self, df: pd.DataFrame):
        """Pre-processes data: resample to 40Hz and scales sensor data"""
        grouped = df.groupby('filename')
        resampled_df = []

        for _, group in grouped:
            group_resampled = self._resample_data(group)
            resampled_df.append(group_resampled)

        df_resampled = pd.concat(resampled_df)
        df_resampled.dropna(inplace=True)

        #save to preprocessed folder
        os.makedirs('preprocessed', exist_ok=True)
        df_resampled.to_csv(os.path.join('preprocessed', f'preprocessed_w{self.window_size}_o{self.overlap}.csv'), index=False)
        return df_resampled

    def _resample_data(self, group: pd.DataFrame):
        """Resample the data to 40Hz (25ms interval) while handling duplicates and non-numeric columns."""

        group['time'] = pd.to_datetime(group['time'], unit='s')  
        group = group.drop_duplicates(subset=['time'])
        group.set_index('time', inplace=True)
        numeric_group = group.drop(columns=['filename'])
        numeric_resampled = numeric_group.resample('20ms').interpolate(method='linear').bfill().ffill()
        numeric_resampled.reset_index(inplace=True)
        numeric_resampled['filename'] = group['filename'].iloc[0]
        return numeric_resampled


    def _classify_fall(self, df: pd.DataFrame):
        #Get 50Mhz data and UMA data
        df_uma = df[df['filename'].str.contains('UMA')]
        df_50mhz = df[~df['filename'].str.contains('UMA')]

        df_uma = self._classify_umafall(df_uma)
        exit()
        df_50mhz = self._classify_50mhz(df_50mhz)
        df = pd.concat([df_uma, df_50mhz])


    def _classify_umafall(self, df: pd.DataFrame):
        df = df.copy()  # Ensure df is not a view
        df['is_fall'] = 0  
        df['time_start'] = pd.to_numeric(df['time_start'], errors='coerce')  # Ensure numeric type
        df = df.dropna(subset=['time_start', 'accel_magnitude', 'gyro_magnitude'])  # Remove NaN rows
        df['time_start'] = df['time_start'].astype('float64') / 10**9  # Convert to seconds

        def extract_first_value(cell):
            """Ensure that each cell is a single float."""
            if isinstance(cell, pd.Series):  # Handle Series inside a cell
                cell = cell.iloc[0]  
            if isinstance(cell, (list, np.ndarray)):  # If it's a list/array, take the first value
                return float(cell[0])
            return float(cell)  # Convert directly if already a scalar
                    
        df['accel_magnitude_sma'] = df['accel_magnitude'].apply(extract_first_value)
        df['gyro_magnitude_sma'] = df['gyro_magnitude_sma'].apply(extract_first_value)
        
        # Normalize accel and gyro
        scaler = MinMaxScaler()
        df[['accel_magnitude_sma', 'gyro_magnitude_sma']] = scaler.fit_transform(df[['accel_magnitude_sma', 'gyro_magnitude_sma']])
        grouped = df.groupby('filename')
        
        for filename, group in grouped:
            time_vals = group['time_start'].values
            accel_vals = group['accel_magnitude_sma'].values
            gyro_vals = group['gyro_magnitude_sma'].values
            
            # Apply heavy smoothing to both accel and gyro values using Gaussian filter
            smoothed_accel = gaussian_filter1d(accel_vals, sigma=5)  # Adjust sigma for smoothing
            smoothed_gyro = gaussian_filter1d(gyro_vals, sigma=5)    # Adjust sigma for smoothing
            
            # Find the index of largest change in both accel and gyro
            accel_diff = np.diff(smoothed_accel)
            gyro_diff = np.diff(smoothed_gyro)
            
            accel_peak_idx = np.argmax(np.abs(accel_diff))  # Largest change in accel
            gyro_peak_idx = np.argmax(np.abs(gyro_diff))    # Largest change in gyro
            
            # Calculate dynamic padding based on the largest change
            max_change_rate = max(np.max(np.abs(accel_diff)), np.max(np.abs(gyro_diff)))
            fall_padding_time = max(1.0, 2.0 * max_change_rate)  # Dynamic padding
            
            # Find the start time by using the index of the largest change
            smallest_idx = accel_peak_idx if accel_peak_idx < gyro_peak_idx else gyro_peak_idx
            start_time = max(time_vals[smallest_idx] - fall_padding_time, time_vals[0])
            
            # Define a threshold for stabilization (e.g., when the rate of change becomes very small)
            stabilization_threshold = 0.01  # Rate of change small enough to consider stable
            
            # Use a while loop to check if the signals have stabilized
            is_stable = False
            end_time = time_vals[-1]  # Start with the last time value
            
            while not is_stable:
                # Check if both accel and gyro are stable (rate of change small)
                accel_rate_of_change = np.max(np.abs(np.diff(smoothed_accel)))  # Maximum rate of change in accel
                gyro_rate_of_change = np.max(np.abs(np.diff(smoothed_gyro)))    # Maximum rate of change in gyro
                
                # If both signals have stabilized (below threshold), break the loop
                if accel_rate_of_change < stabilization_threshold and gyro_rate_of_change < stabilization_threshold:
                    is_stable = True
                else:
                    # If not stable, extend the end time and smooth again
                    fall_padding_time += 0.5  # Extend the fall window
                    end_time = min(time_vals[-1], time_vals[smallest_idx] + fall_padding_time)
                    
                    # Reapply smoothing to get more accurate stabilization
                    smoothed_accel = gaussian_filter1d(accel_vals[:int(end_time)], sigma=5)
                    smoothed_gyro = gaussian_filter1d(gyro_vals[:int(end_time)], sigma=5)
            
            # Plot the magnitudes and detected fall time range
            plt.figure(figsize=(12, 5))
            plt.plot(time_vals, accel_vals, label='Accel Magnitude SMA', color='blue')
            plt.plot(time_vals, gyro_vals, label='Gyro Magnitude SMA', color='orange')
            plt.axvline(start_time, color='red', linestyle='--', label='Start Time')
            plt.axvline(end_time, color='green', linestyle='--', label='End Time')
            plt.axvspan(start_time, end_time, color='red', alpha=0.3, label='Fall')

            plt.legend()
            plt.title(f'{filename}')
            plt.xlabel('Time (s)')
            plt.ylabel('Magnitude')
            plt.grid(True)
            plt.show()

        return df



    
    def _classify_50mhz(self, df: pd.DataFrame):
        """Classify falls for 50Mhz data, based on the 'start_time' and 'end_time' columns"""
        df = df.copy()  # Ensure df is not a view
        df['is_fall'] = 0  # Initialize the 'is_fall' column with 0 (non-fall)
        fall_dict = self._create_fall_dict() 

        for index, row in df.iterrows():
            filename = row['filename']
            start_time, end_time = fall_dict.get(filename, (None, None))  
            if start_time is not None and end_time is not None:
                if start_time <= row['start_time'] <= end_time or start_time <= row['end_time'] <= end_time:
                    df.loc[index, 'is_fall'] = 1  

        return df



    def extract_features(self, df: pd.DataFrame):
        """Applies sliding window and extracts statistical features for each filename"""
        window_size = self.window_size
        step_size = window_size - self.overlap
        grouped = df.groupby('filename')
        features = []
        columns_to_use = ['gyro_x_list', 'gyro_y_list', 'gyro_z_list', 
                        'accel_x_list', 'accel_y_list', 'accel_z_list']

        for filename, group in grouped:
            # Apply sliding window for each group
            for i in range(0, len(group) - window_size + 1, step_size):
                window = group.iloc[i:i + window_size]
                feature_dict = {
                    'filename': filename,
                    'time_start': window['time'].iloc[0],
                    'time_end': window['time'].iloc[-1]
                }

                for col in columns_to_use:
                    feature_dict[f'{col}_mean'] = window[col].mean()
                    feature_dict[f'{col}_std'] = window[col].std()
                    feature_dict[f'{col}_min'] = window[col].min()
                    feature_dict[f'{col}_max'] = window[col].max()
                    feature_dict[f'{col}_median'] = window[col].median()
                    feature_dict[f'{col}_iqr'] = window[col].quantile(0.75) - window[col].quantile(0.25)
                    feature_dict[f'{col}_energy'] = np.sum(window[col]**2)
                    feature_dict[f'{col}_rms'] = np.sqrt(np.mean(window[col]**2)) 
                    sma = np.sum(np.abs(window[col])) / window_size  
                    feature_dict[f'{col}_sma'] = sma
              
                # Calculate Roll and Pitch using Sensor Fusion (Complementary Filter)
                accel_x = window['accel_x_list'].values
                accel_y = window['accel_y_list'].values
                accel_z = window['accel_z_list'].values
                gyro_x = window['gyro_x_list'].values
                gyro_y = window['gyro_y_list'].values
                gyro_z = window['gyro_z_list'].values
                time_diff = 0.02  # Assuming 50Hz sampling rate
                
                roll_accel = np.arctan2(accel_y, np.sqrt(accel_x**2 + accel_z**2)) * (180.0 / np.pi)
                pitch_accel = np.arctan2(-accel_x, np.sqrt(accel_y**2 + accel_z**2)) * (180.0 / np.pi)
                
                roll_gyro = np.cumsum(gyro_x * time_diff)
                pitch_gyro = np.cumsum(gyro_y * time_diff)
                yaw_gyro = np.cumsum(gyro_z * time_diff)
                
                roll_fused = self.alpha * (roll_gyro) + (1 - self.alpha) * roll_accel
                pitch_fused = self.alpha * (pitch_gyro) + (1 - self.alpha) * pitch_accel
                
                feature_dict['roll_mean'] = roll_fused.mean()
                feature_dict['roll_std'] = roll_fused.std()
                feature_dict['pitch_mean'] = pitch_fused.mean()
                feature_dict['pitch_std'] = pitch_fused.std()
                feature_dict['yaw_mean'] = yaw_gyro.mean()
                feature_dict['yaw_std'] = yaw_gyro.std()

                # Compute accel magnitude
                accel_magnitude = np.sqrt(window['accel_x_list']**2 + window['accel_y_list']**2 + window['accel_z_list']**2)
                feature_dict['accel_magnitude'] = accel_magnitude
                feature_dict['accel_magnitude_mean'] = accel_magnitude.mean()
                feature_dict['accel_magnitude_std'] = accel_magnitude.std()
                feature_dict['accel_magnitude_min'] = accel_magnitude.min()
                feature_dict['accel_magnitude_max'] = accel_magnitude.max()
                feature_dict['accel_magnitude_median'] = accel_magnitude.median()
                feature_dict['accel_magnitude_iqr'] = accel_magnitude.quantile(0.75) - accel_magnitude.quantile(0.25)
                feature_dict['accel_magnitude_energy'] = np.sum(accel_magnitude**2)
                feature_dict['accel_magnitude_rms'] = np.sqrt(np.mean(accel_magnitude**2))
                feature_dict['accel_magnitude_sma'] = np.sum(np.abs(accel_magnitude)) / window_size

                # Compute gyro magnitude
                gyro_magnitude = np.sqrt(window['gyro_x_list']**2 + window['gyro_y_list']**2 + window['gyro_z_list']**2)
                feature_dict['gyro_magnitude'] = gyro_magnitude
                feature_dict['gyro_magnitude_mean'] = gyro_magnitude.mean()
                feature_dict['gyro_magnitude_std'] = gyro_magnitude.std()
                feature_dict['gyro_magnitude_min'] = gyro_magnitude.min()
                feature_dict['gyro_magnitude_max'] = gyro_magnitude.max()
                feature_dict['gyro_magnitude_median'] = gyro_magnitude.median()
                feature_dict['gyro_magnitude_iqr'] = gyro_magnitude.quantile(0.75) - gyro_magnitude.quantile(0.25)
                feature_dict['gyro_magnitude_energy'] = np.sum(gyro_magnitude**2)
                feature_dict['gyro_magnitude_rms'] = np.sqrt(np.mean(gyro_magnitude**2))
                feature_dict['gyro_magnitude_sma'] = np.sum(np.abs(gyro_magnitude)) / window_size
                features.append(feature_dict)

        features_df = pd.DataFrame(features)
        features_df = self._classify_fall(features_df)
        # Save to features folder
        os.makedirs('features', exist_ok=True)
        features_df.to_csv(self.get_file_path(), index=False)
        return features_df
    
    def get_file_path(self) -> StrPath:
        """Returns the path to the features file"""
        os.makedirs('features', exist_ok=True)
        return os.path.join('features', f'features_w{self.window_size}_o{self.overlap}.csv')
    
    def _create_fall_dict(self):
        """Creates a dictionary mapping filenames to their start and end times for falls"""
        fall_dict = {}
        for _, row in self.fall_timestamps.iterrows():
            filename = row['filename']
            fall_dict[filename] = (float(row['start_time']), float(row['end_time']))
        
        return fall_dict