import os 
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler

class BaseLoader(ABC):
    def __init__(self, folder_path: str, timestamp_file: str):
        self.datafolder = folder_path
        self.timestamps = pd.read_csv(os.path.join(timestamp_file))

    @abstractmethod
    def load_data(self):
        pass


class UpFallLoader(BaseLoader):
    """
    Loader for the Up Fall dataset
    """
    def __init__(self, folder_path: str, timestamp_file: str):
        super().__init__(folder_path, timestamp_file)

    def load_data(self):
        df = pd.DataFrame()
        falls = os.path.join(self.datafolder, 'Falls')
        adls = os.path.join(self.datafolder, 'ADL')
        fall_timestamps = []
        
        # Load falls data
        for file in os.listdir(falls):
            file_path = os.path.join(falls, file)
            up_df = pd.read_csv(file_path)
            up_df['filename'] = file  # Add a column for the filename
            df = pd.concat([df, up_df], ignore_index=True)
            fall_timestamps.append((file, 0,0))

        #Save fall_start and fall_end timestamps to a csv file
        fall_timestamps_df = pd.DataFrame(fall_timestamps, columns=['filename', 'fall_start', 'fall_end'])
        fall_timestamps_df.to_csv('UP_fall_timestamps.csv', index=False)
        
        # Load ADL data
        for file in os.listdir(adls):
            file_path = os.path.join(adls, file)
            up_df = pd.read_csv(file_path)
            up_df['filename'] = file  # Add a column for the filename
            df = pd.concat([df, up_df], ignore_index=True)

        # Ensure 'time' column is in datetime format
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = df.groupby('filename')['time'].transform(lambda x: (x - x.iloc[0]).dt.total_seconds()).astype(float)

        # Merge with timestamps
        df = pd.merge(df, self.timestamps, how='left', left_on='filename', right_on='filename')
        print(df)
        return df








class WedaFallLoader(BaseLoader):
    """
    Loader for the Weda Fall dataset
    """
    def __init__(self, folder_path: str, timestamp_file: str):
        super().__init__(folder_path, timestamp_file)

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

        #Create fall_start and fall_end columns based on timestamps csv using filename column as key
        df = pd.merge(df, self.timestamps, how='left', left_on='filename', right_on='filename')
        return df



class UmaFallLoader(BaseLoader):
    """
    Loader for the Uma Fall dataset
    """
    def __init__(self, folder_path: str, timestamp_file: str):
        super().__init__(folder_path, timestamp_file)

    def load_data(self):
        df = pd.DataFrame()
        WRIST_SENSOR_ID = 3
        ACCELERO_SENSOR_TYPE = 0
        GYRO_SENSOR_TYPE = 1

        falls = os.path.join(self.datafolder, 'Falls')
        adl = os.path.join(self.datafolder, 'ADL')

        for file in os.listdir(falls):
            file_path = os.path.join(falls, file)
            
            # Read the CSV and process
            uma_df = pd.read_csv(file_path, skiprows=40, sep=';') 
            uma_df.columns = uma_df.columns.str.strip()
            uma_df.dropna(axis=1, how='all', inplace=True)

            

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

        for file in os.listdir(adl):
            file_path = os.path.join(adl, file)
            
            # Read the CSV and process
            uma_df = pd.read_csv(file_path, skiprows=40, sep=';') 
            uma_df.columns = uma_df.columns.str.strip()
            uma_df.dropna(axis=1, how='all', inplace=True)

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
            df = pd.concat([df,uma_df])
        
        #Create fall_start and fall_end columns based on timestamps csv using filename column as key
        df = pd.merge(df, self.timestamps, how='left', left_on='filename', right_on='filename')

        return df
    

