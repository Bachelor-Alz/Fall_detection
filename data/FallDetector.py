import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Union
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message=".*Precision loss occurred.*")


StrPath = Union[str, os.PathLike]

class FallDetector:
    def __init__(self, window_size: int, overlap: int, datafolder: StrPath):
        self.window_size = window_size
        self.overlap = overlap
        self.fall_timestamps = pd.read_csv('fall_timestamps.csv')
        self.datafolder = datafolder

    def load_data(self):
        folders = os.listdir(self.datafolder)
        sorted_folders = sorted(folders)
        df = pd.DataFrame()

        # 50 MHz
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


        # UMA Fall
        uma_data = os.path.join(os.getcwd(), 'UMAFall')

        for file in os.listdir(uma_data):
            file_path = os.path.join(uma_data, file)
            
            # Read the CSV and process
            uma_df = pd.read_csv(file_path, skiprows=40, sep=';') 
            uma_df.columns = uma_df.columns.str.strip()
            uma_df.dropna(axis=1, how='all', inplace=True)
            WRIST_SENSOR_ID = 3
            ACCELERO_SENSOR_TYPE = 0
            GYRO_SENSOR_TYPE = 1

            uma_df = uma_df[uma_df['Sensor ID'] == WRIST_SENSOR_ID]
            uma_df = uma_df[uma_df['Sensor Type'].isin([GYRO_SENSOR_TYPE, ACCELERO_SENSOR_TYPE])]
            uma_df = uma_df.rename(columns={'% TimeStamp': 'time'})
            
            # Convert timestamp in ms to seconds
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

        #save to preprocessed folder
        os.makedirs('preprocessed', exist_ok=True)
        df_resampled.to_csv(os.path.join('preprocessed', f'preprocessed_w{self.window_size}_o{self.overlap}.csv'), index=False)
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

        group['time'] = pd.to_datetime(group['time'], unit='s')  
        group = group.drop_duplicates(subset=['time'])
        group.set_index('time', inplace=True)
        numeric_group = group.drop(columns=['filename'])
        new_start_time = numeric_group.index.min().floor('20ms')  # Align start time to nearest 20ms
        new_time_index = pd.date_range(start=new_start_time, 
                                    end=numeric_group.index.max(), 
                                    freq='20ms')

        numeric_resampled = numeric_group.reindex(new_time_index).interpolate(method='linear').bfill().ffill()
        numeric_resampled.reset_index(inplace=True)
        numeric_resampled.rename(columns={'index': 'time'}, inplace=True)
        numeric_resampled['filename'] = group['filename'].iloc[0]

        return numeric_resampled



    def _classify_fall(self, df: pd.DataFrame):
        #Get 50Mhz data and UMA data
        df['is_fall'] = 0
        df_uma = df[df['filename'].str.contains('UMA')]
        df_50mhz = df[~df['filename'].str.contains('UMA')]
        df_uma = self._classify_umafall(df_uma)
        df_50mhz = self._classify_50mhz(df_50mhz)
        df = pd.concat([df_uma, df_50mhz])
        return df

    def _classify_umafall(self, df: pd.DataFrame):
        # Keep the original dataframe for later
        og_df = df.copy()
        scaler = MinMaxScaler()

        df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce')
        df['end_time'] = pd.to_numeric(df['end_time'], errors='coerce')  # Ensure end_time is numeric
        df['start_time_sec'] = df['start_time'] / 10**9  # Assuming start_time is in nanoseconds
        df['end_time_sec'] = df['end_time'] / 10**9  # Convert end_time to seconds as well
        df[['accel_magnitude_sma', 'gyro_magnitude_sma']] = scaler.fit_transform(df[['accel_magnitude_sma', 'gyro_magnitude_sma']])

        fall_indices = []

        # Ensure the file is created if it doesn't exist already
        if not os.path.exists('UMA_fall_timestamps.csv'):
            fall_columns = ['filename', 'start_time', 'end_time']
            pd.DataFrame(columns=fall_columns).to_csv('UMA_fall_timestamps.csv', index=False)

        #print how many groups are in the dataframe
        print(f"Number of groups in dataframe: {len(df.groupby('filename'))}")

        for filename, group in df.groupby('filename'):
            #if the filename is already in UMA_fall_timestamps.csv, skip it
            if filename in pd.read_csv('UMA_fall_timestamps.csv')['filename'].values:
                continue

            time_vals = group['start_time_sec'].values
            accel_vals = group['accel_magnitude_sma'].values
            gyro_vals = group['gyro_magnitude_sma'].values

            accel_diff = np.diff(accel_vals)
            gyro_diff = np.diff(gyro_vals)

            change_threshold = 0.01  # Threshold for significant change
            no_change_duration_limit = 1.5  # Max seconds without change

            # Detect the start of a fall
            start_idx = next((i for i in range(1, len(accel_diff)) if 
                            abs(accel_diff[i] - accel_diff[i-1]) > change_threshold or 
                            abs(gyro_diff[i] - gyro_diff[i-1]) > change_threshold), None)

            if start_idx is None:
                start_idx = 0

            start_time = max(time_vals[start_idx] - 1, 0)
            end_time = start_time
            last_change_time = start_time

            # Detect the end of the fall
            for i in range(start_idx, len(time_vals) - 1):
                if abs(accel_diff[i]) > change_threshold or abs(gyro_diff[i]) > change_threshold:
                    end_time = time_vals[i]
                    last_change_time = end_time
                elif time_vals[i] - last_change_time > no_change_duration_limit:
                    break  # End the event if no significant change for too long

            for idx, row in group.iterrows():
                # Direct comparison with seconds-based timestamp
                if start_time <= row['start_time_sec'] <= end_time or start_time <= row['end_time_sec'] <= end_time:
                    fall_indices.append(idx)

            # Plot fall detection
            results = self.plot_fall_detection(group['filename'].iloc[0], time_vals, accel_vals, gyro_vals, start_time, end_time)
            pd.DataFrame([results]).to_csv('UMA_fall_timestamps.csv', mode='a', header=False, index=False)

        # Assign 'is_fall' flag to the original dataframe based on detected fall indices
        og_df.loc[fall_indices, 'is_fall'] = 1
        return og_df


    def plot_fall_detection(self, filename, time_vals, accel_vals, gyro_vals, start_time, end_time):
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
        start_time, end_time = map(float, input(f"Enter fall start and end time for {filename} (separated by space): ").split())
        return {'filename': filename, 'start_time': start_time, 'end_time': end_time}

        
    def _classify_50mhz(self, df: pd.DataFrame):
        """Classify falls for 50Mhz data, based on the 'start_time' and 'end_time' columns"""
        df = df.copy()  # Ensure df is not a view
        fall_dict = self._create_fall_dict()

        for index, row in df.iterrows():
            filename = row['filename']
            start_time, end_time = fall_dict.get(filename, (None, None))

            if start_time is not None and end_time is not None:
                row_start_time = row['start_time'].timestamp()
                row_end_time = row['end_time'].timestamp()

                if start_time <= row_start_time <= end_time or start_time <= row_end_time <= end_time:
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
                    'start_time': window['time'].iloc[0],
                    'end_time': window['time'].iloc[-1]
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
        features_df = self._classify_fall(features_df)
        cols = ['is_fall'] + [col for col in features_df.columns if col != 'is_fall']
        features_df = features_df[cols]

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