import pandas as pd
import numpy as np
import os
from typing import Union
from sklearn.preprocessing import StandardScaler
import warnings
from LoadData import BaseLoader
from concurrent.futures import ThreadPoolExecutor

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
    
    def pre_process(self, df: pd.DataFrame):
        """Pre-processes data: resample to 50Hz and scales sensor data"""
        #Group my UMA and 50Mhz data
        uma_df = df[df['filename'].str.contains('UMA', na=False)]
        non_uma_df = df[~df['filename'].str.contains('UMA', na=False)]

        scaler_uma = StandardScaler()
        scaler_non_uma = StandardScaler()

        columns_to_scale = ['accel_x_list', 'accel_y_list', 'accel_z_list',
                            'gyro_x_list', 'gyro_y_list', 'gyro_z_list']

        uma_df[columns_to_scale] = scaler_uma.fit_transform(uma_df[columns_to_scale])
        non_uma_df[columns_to_scale] = scaler_non_uma.fit_transform(non_uma_df[columns_to_scale])
        df = pd.concat([uma_df, non_uma_df])

        df = self._remove_outliers_iqr(df)
        grouped = df.groupby('filename')
        resampled_df = []

        for _, group in grouped:
            group_resampled = self._resample_data(group)
            resampled_df.append(group_resampled)

        df_resampled = pd.concat(resampled_df)
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
        numeric_group = group.drop(columns=['filename'])

        # Align start time to nearest 20ms
        new_start_time = numeric_group.index.min().floor('20ms')  
        new_time_index = pd.date_range(start=new_start_time, 
                                    end=numeric_group.index.max(), 
                                    freq='20ms')

        numeric_resampled = numeric_group.reindex(new_time_index).interpolate(method='linear').bfill().ffill()
        numeric_resampled.reset_index(inplace=True)
        numeric_resampled.rename(columns={'index': 'time'}, inplace=True)
        numeric_resampled['filename'] = group['filename'].iloc[0]
        return numeric_resampled

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

                # Check if either start_time or end_time is NaN, and label as 0 if so
                if window['start_time'].isna().any() or window['end_time'].isna().any():
                    is_fall = 0
                else:
                    
                    fall_samples_in_window = ((window['time'] >= window['start_time']) & 
                                            (window['time'] <= window['end_time'])).sum()
                    is_fall = 1 if fall_samples_in_window / window_size >= 0.5 else 0

                feature_dict = {
                    'filename': filename,
                    'start_time': window['time'].iloc[0],
                    'end_time': window['time'].iloc[-1],
                    'is_fall': is_fall
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
                    feature_dict[f'{col}_sma'] = np.sum(np.abs(window[col])) / window_size  
                    feature_dict[f'{col}_skew'] = window[col].skew()
                    feature_dict[f'{col}_kurtosis'] = window[col].kurtosis()
                    feature_dict[f'{col}_absum'] = np.sum(np.abs(window[col]))

                    feature_dict[f'{col}_DDM']  = feature_dict[f'{col}_max'] - feature_dict[f'{col}_min']
                    feature_dict[f'{col}_GMM'] = np.sqrt((feature_dict[f'{col}_max'] - feature_dict[f'{col}_min'])**2 + (np.argmax(feature_dict[f'{col}_max']) - np.argmax(feature_dict[f'{col}_min']))**2)
                    feature_dict[f'{col}_MD'] = max(np.diff(window[col]))

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

                gyro_magnitude = np.sqrt(window['gyro_x_list']**2 + window['gyro_y_list']**2 + window['gyro_z_list']**2)
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

        # Save to features folder
        os.makedirs('features', exist_ok=True)
        features_df.to_csv(self.get_file_path(), index=False)
        return features_df
    
    def get_file_path(self) -> StrPath:
        """Returns the path to the features file"""
        os.makedirs('features', exist_ok=True)
        return os.path.join('features', f'features_w{self.window_size}_o{self.overlap}.csv')
    