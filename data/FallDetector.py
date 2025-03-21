import re
import pandas as pd
import numpy as np
import os
from typing import Union
from sklearn.preprocessing import StandardScaler
import warnings
from LoadData import BaseLoader
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

warnings.filterwarnings('ignore', category=RuntimeWarning, message=".*Precision loss occurred.*")
StrPath = Union[str, os.PathLike]

class FallDetector:
    def __init__(self, window_size: int, overlap: int, data_loaders: list[BaseLoader]):
        self.window_size = window_size
        self.overlap = overlap
        self.data_loaders = data_loaders


    def load_data(self):
        """Loads data from all data loaders"""

        def load_single_loader(loader):
            return loader.load_data()

        with ThreadPoolExecutor() as executor:
            data_frames = list(executor.map(load_single_loader, self.data_loaders))

        
        # Combine all loaded data
        df = pd.concat(data_frames)

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
        """Pre-processes data: resample to 50Hz and scales sensor data"""
        #Group my UMA and 50Mhz data
        #uma_df = df[df['filename'].str.contains('UMA', na=False)]
        non_uma_df = df[~df['filename'].str.contains('UMA', na=False)]

        scaler_uma = StandardScaler()
        scaler_non_uma = StandardScaler()

        columns_to_scale = ['accel_x_list', 'accel_y_list', 'accel_z_list',
                            'gyro_x_list', 'gyro_y_list', 'gyro_z_list']
        
        for col in columns_to_scale:
            # Fix multiple periods in string values
            #uma_df[col] = uma_df[col].apply(self.fix_multiple_periods)
            non_uma_df[col] = non_uma_df[col].apply(self.fix_multiple_periods)

            # Convert to numeric, replacing errors with NaN
            #uma_df[col] = pd.to_numeric(uma_df[col], errors='coerce')
            non_uma_df[col] = pd.to_numeric(non_uma_df[col], errors='coerce')


        #uma_df[columns_to_scale] = scaler_uma.fit_transform(uma_df[columns_to_scale])
        non_uma_df[columns_to_scale] = scaler_non_uma.fit_transform(non_uma_df[columns_to_scale])
        
        #df = pd.concat([uma_df, non_uma_df])

        #TODO GO BACK TO DF 
        df = self._remove_outliers_iqr(non_uma_df)
        print(df)
        grouped = df.groupby('filename')
        resampled_df = []

        for _, group in grouped:
            group_resampled = self._resample_data(group)
            resampled_df.append(group_resampled)
        

        df_resampled = pd.concat(resampled_df)
        print(df_resampled)
        df_resampled.dropna(inplace=True)
        
        return df_resampled
    

    def _remove_outliers_iqr(self, df: pd.DataFrame):
        # Drop 'filename' column for IQR calculation
        df_numeric = df.drop(columns='filename')
        Q1 = df_numeric.quantile(0.25)
        Q3 = df_numeric.quantile(0.75)
        IQR = Q3 - Q1

        # Filter out rows with outliers
        filtered_df = df[~((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).any(axis=1)]
        return filtered_df

    def _resample_data(self, group: pd.DataFrame):
        """Resample the data to 50Hz (20ms interval) while handling duplicates and non-numeric columns.
        Also aligns the time to start at 00:00.000.
        """  
        group = group.drop_duplicates(subset=['time'])
        group.set_index('time', inplace=True)

        # Snap timestamps to the nearest 20ms
        group.index = group.index.round('20ms')

        # Drop the non-numeric column 'filename'
        numeric_group = group.drop(columns=['filename'])

        # Resample the numeric data to 20ms intervals
        numeric_resampled = numeric_group.resample('20ms')

        # Interpolate missing values using linear interpolation, then backfill and forward fill
        interpolated = numeric_resampled.interpolate(method='linear').bfill().ffill()

        # Reset the index and rename it back to 'time'
        interpolated.reset_index(inplace=True)
        interpolated.rename(columns={'index': 'time'}, inplace=True)

        # Add the 'filename' column back
        interpolated['filename'] = group['filename'].iloc[0]

        # Return the final interpolated DataFrame
        return interpolated


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
                    if not (window['start_time'].isna().any() or window['end_time'].isna().any()):
                        futures.append(executor.submit(FallDetector.process_window, window, filename, window_size))

            for future in futures:
                features.append(future.result())

        features_df = pd.DataFrame(features)
        os.makedirs('features', exist_ok=True)
        features_df.to_csv(self.get_file_path(), index=False)
        print(features_df)
        return features_df
    
    @staticmethod
    def process_window(window, filename, window_size):
            feature_dict = {
                'filename': filename,
                'start_time': window['time'].iloc[0],
                'end_time': window['time'].iloc[-1],
                'is_fall': 1 if ((window['time'] >= window['start_time']) & 
                                (window['time'] <= window['end_time'])).sum() / window_size >= 0.5 else 0
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
    