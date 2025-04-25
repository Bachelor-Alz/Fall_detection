import re
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from process_window import process_window
import joblib
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

        df = pd.concat(data_frames)
        df['start_time'] = pd.to_datetime(df['start_time'], errors="coerce", unit='s').fillna(pd.NaT)
        df['end_time'] = pd.to_datetime(df['end_time'], errors="coerce", unit='s').fillna(pd.NaT)
        df['time'] = pd.to_datetime(df['time'], errors="coerce", unit='s')
        print("Loaded data")
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
            

        # Scale the individual datasets first
        scaler_uma = StandardScaler()
        scaler_non_uma = StandardScaler()
        scaler_up = StandardScaler()

        uma_df[columns_to_scale] = scaler_uma.fit_transform(uma_df[columns_to_scale])
        weda_df[columns_to_scale] = scaler_non_uma.fit_transform(weda_df[columns_to_scale])
        up_df[columns_to_scale] = scaler_up.fit_transform(up_df[columns_to_scale])

        # Combine datasets after individual scaling
        df_combined = pd.concat([uma_df, weda_df, up_df])

        # Now, scale the entire combined dataset (optional collective scaling step)
        scaler_combined = StandardScaler()
        df_combined[columns_to_scale] = scaler_combined.fit_transform(df_combined[columns_to_scale])

        # Save the combined scaler for later use in the API
        save_location = os.path.join(os.pardir, 'combined_scaler.joblib')
        joblib.dump(scaler_combined, save_location)

        # Resample data
        grouped = df_combined.groupby('filename')
        resampled_df = []

        for _, group in grouped:
            group_resampled = self._resample_data(group)
            resampled_df.append(group_resampled)

        df_resampled = pd.concat(resampled_df)

        # Apply Gaussian smoothing per filename
        smoothed_df = []
        for _, group in df_resampled.groupby('filename'):
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
        grouped = df.groupby('filename')
        features = []
        step_size = self.window_size - self.overlap  

        excluded_features = self._get_excluded_features()

        with ProcessPoolExecutor() as executor:
            futures = []
            for filename, group in grouped:
                futures.append(executor.submit(process_window, group, self.window_size, step_size, include_metadata=True))

            for future in futures:
                features.append(future.result())

        features_df = pd.concat(features)
        os.makedirs('features', exist_ok=True)
        features_df.to_csv(self.get_file_path(), index=False)
        print("Extracted features")
        return features_df
    
    def _get_excluded_features(self):
        # Read PCA feature importance for this window size and overlap
        pca_file_path = os.path.join('pca_feature_importance', f'feature_importance_w{self.window_size}_o{self.overlap}.csv')
        if not os.path.exists(pca_file_path):
            return None

        df = pd.read_csv(pca_file_path)
        excluded_features = set(df[df['PCA_Weighted_Importance'] < 0.01]['Feature'].tolist())
        return excluded_features

    def get_file_path(self) -> StrPath:
        """Returns the path to the features file"""
        os.makedirs('features', exist_ok=True)
        return os.path.join('features', f'features_w{self.window_size}_o{self.overlap}.csv')
    
