import pandas as pd
import numpy as np
import os
from typing import Union
from scipy.stats import kurtosis, skew
from sklearn.discriminant_analysis import StandardScaler
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

        return df
    
    def pre_process(self, df: pd.DataFrame):
        """Pre-processes data: resample to 40Hz and scales sensor data"""
        grouped = df.groupby('filename')
        resampled_df = []

        for _, group in grouped:
            group_resampled = self._resample_data(group)
            resampled_df.append(group_resampled)

        df_resampled = pd.concat(resampled_df)

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
        numeric_resampled = numeric_group.resample('20ms').interpolate(method='linear')
        numeric_resampled['filename'] = group['filename'].iloc[0]
        return numeric_resampled.reset_index()


    
    def extract_features(self, df: pd.DataFrame):
        """Applies sliding window and extracts statistical features for each filename"""
        window_size = self.window_size
        step_size = window_size - self.overlap
        fall_dict = self._create_fall_dict()  # Assuming this is a dictionary with filenames and their fall start/end times
        grouped = df.groupby('filename')
        features = []
        columns_to_use = ['gyro_x_list', 'gyro_y_list', 'gyro_z_list', 
                        'accel_x_list', 'accel_y_list', 'accel_z_list']

        for filename, group in grouped:
            # Get fall start and end times for the current filename
            fall_start, fall_end = fall_dict.get(filename, (None, None))

            # Apply sliding window for each group
            for i in range(0, len(group) - window_size + 1, step_size):
                window = group.iloc[i:i + window_size]
                feature_dict = {
                    'is_fall': 1 if self.is_within_fall(fall_start, fall_end, window) else 0,
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
                feature_dict['accel_magnitude_mean'] = accel_magnitude.mean()
                feature_dict['accel_magnitude_std'] = accel_magnitude.std()
                feature_dict['accel_magnitude_min'] = accel_magnitude.min()
                feature_dict['accel_magnitude_max'] = accel_magnitude.max()
                feature_dict['accel_magnitude_median'] = accel_magnitude.median()
                feature_dict['accel_magnitude_iqr'] = accel_magnitude.quantile(0.75) - accel_magnitude.quantile(0.25)
                feature_dict['accel_magnitude_energy'] = np.sum(accel_magnitude**2)
                feature_dict['accel_magnitude_rms'] = np.sqrt(np.mean(accel_magnitude**2))
                feature_dict['accel_magnitude_sma'] = np.sum(np.abs(accel_magnitude)) / window_size
                            
                features.append(feature_dict)

        features_df = pd.DataFrame(features)
        # Save to features folder
        os.makedirs('features', exist_ok=True)
        features_df.to_csv(self.get_file_path(), index=False)
        return features_df
    
    def get_file_path(self) -> StrPath:
        """Returns the path to the features file"""
        os.makedirs('features', exist_ok=True)
        return os.path.join('features', f'features_w{self.window_size}_o{self.overlap}.csv')
    
    def is_within_fall(self, fall_start, fall_end, window) -> bool:
        """Check if a fall event happens within the given window"""
        if fall_start is None or fall_end is None:
            return False  
        
        window_start_time = window['time'].iloc[0].timestamp()
        window_end_time = window['time'].iloc[-1].timestamp()
        fall_start = float(fall_start)
        fall_end = float(fall_end)

        return not (window_end_time < fall_start or window_start_time > fall_end)
    
    def _create_fall_dict(self):
        """Creates a dictionary mapping filenames to their start and end times for falls"""
        fall_dict = {}
        for _, row in self.fall_timestamps.iterrows():
            filename = row['filename']
            fall_dict[filename] = (float(row['start_time']), float(row['end_time']))
        
        return fall_dict