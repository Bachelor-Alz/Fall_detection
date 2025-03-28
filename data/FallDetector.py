import re
import pandas as pd
import numpy as np
import os
from typing import Union
from sklearn.preprocessing import StandardScaler
import warnings
from LoadData import BaseLoader
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore', category=RuntimeWarning, message=".*Precision loss occurred.*")
StrPath = Union[str, os.PathLike]

class FallDetector:
    def __init__(self, window_size: int, overlap: int, data_loaders: list[BaseLoader]):
        self.window_size = window_size
        self.overlap = overlap
        self.data_loaders = data_loaders


    def load_data(self):
        """Loads data from all data loaders"""
        print("Loading data")
        def load_single_loader(loader):
            return loader.load_data()

        with ThreadPoolExecutor() as executor:
            data_frames = list(executor.map(load_single_loader, self.data_loaders))

        
        # Combine all loaded data
        df = pd.concat(data_frames)
        print("Loaded data")

        # Common operations for all datasets
        df['start_time'] = pd.to_datetime(df['start_time'], unit='s')
        df['end_time'] = pd.to_datetime(df['end_time'], unit='s')
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def fix_multiple_periods(self, value):
        """Fix values with multiple periods (e.g., '31.203.125' -> '31.203125')"""
        if isinstance(value, str):
            value = re.sub(r'(\d+)\.(\d+)\.(\d+)', r'\1\.\2\3', value)
        return value

    def pre_process(self, df: pd.DataFrame):
        """Pre-processes data: resample to 50Hz, scale sensor data, and apply Gaussian smoothing per filename"""
        print("Pre-processing data")
        uma_df = df[df['filename'].str.contains('UMA', na=False)]
        up_df = df[~df['filename'].str.contains('UMA|U', na=False)]
        weda_df = df[df['filename'].str.contains(r'U\d{2}_R\d{2}', na=False, regex=True)]

        columns_to_scale = ['accel_x_list', 'accel_y_list', 'accel_z_list',
                            'gyro_x_list', 'gyro_y_list', 'gyro_z_list']

        # Fix multiple periods and convert to numeric
        for col in columns_to_scale:
            uma_df.loc[:, col] = uma_df[col].apply(self.fix_multiple_periods)
            weda_df.loc[:, col] = weda_df[col].apply(self.fix_multiple_periods)
            up_df.loc[:, col] = up_df[col].apply(self.fix_multiple_periods)

            uma_df.loc[:, col] = pd.to_numeric(uma_df[col], errors='coerce')
            weda_df.loc[:, col] = pd.to_numeric(weda_df[col], errors='coerce')
            up_df.loc[:, col] = pd.to_numeric(up_df[col], errors='coerce')

        # Scale the data
        scaler_uma = StandardScaler()
        scaler_non_uma = StandardScaler()
        scaler_up = StandardScaler()

        uma_df.loc[:, columns_to_scale] = scaler_uma.fit_transform(uma_df[columns_to_scale])
        weda_df.loc[:, columns_to_scale] = scaler_non_uma.fit_transform(weda_df[columns_to_scale])
        up_df.loc[:, columns_to_scale] = scaler_up.fit_transform(up_df[columns_to_scale])

        # Combine dataframes
        df = pd.concat([uma_df, weda_df, up_df])

        # Resample data
        grouped = df.groupby('filename')
        resampled_df = []

        for _, group in grouped:
            group_resampled = self._resample_data(group)
            resampled_df.append(group_resampled)

        df_resampled = pd.concat(resampled_df)

        # Apply Gaussian smoothing per filename
        smoothed_df = []
        for filename, group in df_resampled.groupby('filename'):
            for col in columns_to_scale:
                group[col] = self._apply_gaussian_filter(group[col].values)
            smoothed_df.append(group)

        df_smoothed = pd.concat(smoothed_df)
        print("Pre-processed data")
        return df_smoothed
   

    def _resample_data(self, group: pd.DataFrame):
        """Resample the data to 50Hz (20ms interval) while handling duplicates and non-numeric columns.
        Also aligns the time to start at 00:00.000.
        """  
        group.set_index('time', inplace=True)
        ms = '20.00ms'
        group.index = group.index.round(ms)
        group = group[~group.index.duplicated(keep='first')]
        
        # Drop the non-numeric column 'filename'
        numeric_group = group.drop(columns=['filename'])
        
        # Separate datetime columns (start_time and end_time) from numeric columns
        datetime_cols = ['start_time', 'end_time']
        datetime_data = numeric_group[datetime_cols]  # Save datetime columns
        numeric_data = numeric_group.drop(columns=datetime_cols)  # Keep only numeric data

        # Ensure numeric columns are converted to numeric types
        numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')

        # Align start time to nearest 20ms
        new_start_time = numeric_group.index.min().floor(ms)  
        new_time_index = pd.date_range(start=new_start_time, 
                                    end=numeric_group.index.max(), 
                                    freq=ms)
        
        # Resample and interpolate only numeric data
        numeric_resampled = numeric_data.reindex(new_time_index).interpolate(method='linear').bfill().ffill()
        
        # Reattach the datetime columns without interpolation
        numeric_resampled[datetime_cols] = datetime_data.reindex(new_time_index, method='ffill')
        
        numeric_resampled.reset_index(inplace=True)
        numeric_resampled.rename(columns={'index': 'time'}, inplace=True)
        numeric_resampled['filename'] = group['filename'].iloc[0]
        return numeric_resampled
    
    def _apply_gaussian_filter(self, data, sigma=5):
        return gaussian_filter1d(data, sigma=sigma)
        


    def extract_features(self, df: pd.DataFrame):
        """Applies sliding window and extracts statistical features for each filename"""
        window_size = self.window_size
        step_size = window_size - self.overlap
        grouped = df.groupby('filename')
        features = []

        with ProcessPoolExecutor() as executor:
            futures = []
            for filename, group in grouped:
                for i in range(0, len(group) - window_size + 1, step_size):
                    window = group.iloc[i:i + window_size]
                    futures.append(executor.submit(FallDetector.process_window, window, filename, window_size))

            for future in futures:
                features.append(future.result())

        features_df = pd.DataFrame(features)
        os.makedirs('features', exist_ok=True)
        features_df.to_csv(self.get_file_path(), index=False)
        print("Extracted features")
        return features_df
    
    @staticmethod
    def process_window(window, filename, window_size):
            if(window['start_time'] is None or window['end_time'] is None):
                is_fall = 0
            else:
                is_fall = 1 if ((window['time'] >= window['start_time']) & 
                                (window['time'] <= window['end_time'])).sum() / window_size >= 0.6 else 0
            
            feature_dict = {
                'filename': filename,
                'start_time': window['time'].iloc[0],
                'end_time': window['time'].iloc[-1],
                'is_fall': is_fall
            }

            columns_to_use = ['gyro_x_list', 'gyro_y_list', 'gyro_z_list', 
                        'accel_x_list', 'accel_y_list', 'accel_z_list']

            for col in columns_to_use:
                col_data = window[col]
                feature_dict[f'{col}_mean'] = col_data.mean()
                feature_dict[f'{col}_std'] = col_data.std()
                feature_dict[f'{col}_min'] = col_data.min()
                feature_dict[f'{col}_max'] = col_data.max()
                feature_dict[f'{col}_median'] = col_data.median()
                feature_dict[f'{col}_iqr'] = col_data.quantile(0.75) - col_data.quantile(0.25)
                feature_dict[f'{col}_energy'] = np.sum(col_data**2)
                feature_dict[f'{col}_rms'] = np.sqrt(np.mean(col_data**2))
                feature_dict[f'{col}_sma'] = np.sum(np.abs(col_data)) / window_size
                feature_dict[f'{col}_skew'] = col_data.skew()
                feature_dict[f'{col}_kurtosis'] = col_data.kurtosis()
                feature_dict[f'{col}_absum'] = np.sum(np.abs(col_data))
                feature_dict[f'{col}_DDM'] = col_data.max() - col_data.min()
                feature_dict[f'{col}_GMM'] = np.sqrt((col_data.max() - col_data.min())**2 + 
                                                    (np.argmax(col_data) - np.argmin(col_data))**2)
                feature_dict[f'{col}_MD'] = max(np.diff(col_data))

            accel_magnitude = np.sqrt(window['accel_x_list']**2 + window['accel_y_list']**2 + window['accel_z_list']**2)
            gyro_magnitude = np.sqrt(window['gyro_x_list']**2 + window['gyro_y_list']**2 + window['gyro_z_list']**2)

            for mag, prefix in zip([accel_magnitude, gyro_magnitude], ['accel_magnitude', 'gyro_magnitude']):
                feature_dict[f'{prefix}_mean'] = mag.mean()
                feature_dict[f'{prefix}_std'] = mag.std()
                feature_dict[f'{prefix}_min'] = mag.min()
                feature_dict[f'{prefix}_max'] = mag.max()
                feature_dict[f'{prefix}_median'] = mag.median()
                feature_dict[f'{prefix}_iqr'] = mag.quantile(0.75) - mag.quantile(0.25)
                feature_dict[f'{prefix}_energy'] = np.sum(mag**2)
                feature_dict[f'{prefix}_rms'] = np.sqrt(np.mean(mag**2))
                feature_dict[f'{prefix}_sma'] = np.sum(np.abs(mag)) / window_size

            return feature_dict
    
    def get_file_path(self) -> StrPath:
        """Returns the path to the features file"""
        os.makedirs('features', exist_ok=True)
        return os.path.join('features', f'features_w{self.window_size}_o{self.overlap}.csv')
    