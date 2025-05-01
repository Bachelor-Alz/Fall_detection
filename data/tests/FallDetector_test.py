# test_fall_detector.py
import sys
import os

import numpy as np

from data.LoadData import BaseLoader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from process_window import process_window

import pandas as pd;
from data.FallDetector import FallDetector


def test_fix_multiple_periods():
    detector = FallDetector(100, 50, [])
    assert detector.fix_multiple_periods("31.203.125") == "31.203125"
    assert detector.fix_multiple_periods("12.34") == "12.34"
    assert detector.fix_multiple_periods("12.34.56.54") == "12.345654"

def test_apply_gaussian_filter():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    detector = FallDetector(window_size=100, overlap=50, data_loaders=[])
    smoothed_data = detector._apply_gaussian_filter(data, sigma=5)
    assert smoothed_data[0] != 1.0 


# Mock class to simulate BaseLoader with direct DataFrame return
class MockLoader(BaseLoader):
    def __init__(self, name: str):
        self.name = name
        self.timestamps = pd.DataFrame({'time': [1, 2, 3], 'start_time': [0, 1, 2], 'end_time': [1, 2, 3]})

    def load_data(self):
        data = {
            'time': [1, 2, 3],
            'start_time': [0, 1, 2],
            'end_time': [1, 2, 3],
            'filename': [self.name, self.name, self.name]
        }
        return pd.DataFrame(data)


def test_load_data():
    loader1 = MockLoader(name='Loader1')
    loader2 = MockLoader(name='Loader2')
    detector = FallDetector(window_size=100, overlap=50, data_loaders=[loader1, loader2])
    result_df = detector.load_data()

    assert len(result_df) == 6  
    assert all(result_df['filename'] == ['Loader1', 'Loader1', 'Loader1', 'Loader2', 'Loader2', 'Loader2'])
    assert 'time' in result_df.columns  
    assert 'start_time' in result_df.columns  
    assert 'end_time' in result_df.columns  

    assert pd.api.types.is_datetime64_any_dtype(result_df['time'])
    assert pd.api.types.is_datetime64_any_dtype(result_df['start_time'])
    assert pd.api.types.is_datetime64_any_dtype(result_df['end_time'])

def test_resample_data_misalignment():
    # Simulate input data
    data = {
        'time': ['2025-05-01 00:00:00.000', '2025-05-01 00:00:00.021', '2025-05-01 00:00:00.041', '2025-05-01 00:00:00.055'],
        'start_time': ['2025-05-01 00:00:00', '2025-05-01 00:00:00', '2025-05-01 00:00:00', '2025-05-01 00:00:00'],
        'end_time': ['2025-05-01 00:00:01', '2025-05-01 00:00:01', '2025-05-01 00:00:01', '2025-05-01 00:00:01'],
        'value1': [1.0, 2.0, 3.0, 4.0],
        'value2': [5.0, 6.0, 7.0, 8.0],
        'filename': ['file1', 'file1', 'file1', 'file1']
    }
    group = pd.DataFrame(data)
    group['time'] = pd.to_datetime(group['time'])
    detector = FallDetector(window_size=100, overlap=50, data_loaders=[])
    result = detector._resample_data(group)
    
    # Ensure the index (time) is rounded to the nearest 20ms.
    assert result['time'][0] == pd.Timestamp('2025-05-01 00:00:00')
    assert result['time'][1] == pd.Timestamp('2025-05-01 00:00:00.020')
    assert result['time'][2] == pd.Timestamp('2025-05-01 00:00:00.040')
    assert result['time'][3] == pd.Timestamp('2025-05-01 00:00:00.060')

def test_resample_data_with_gaps():
    data = {
        'time': ['2025-05-01 00:00:00.000', '2025-05-01 00:00:01.500', '2025-05-01 00:00:03.000'], 
        'start_time': ['2025-05-01 00:00:00', '2025-05-01 00:00:00', '2025-05-01 00:00:00'],
        'end_time': ['2025-05-01 00:00:01', '2025-05-01 00:00:02', '2025-05-01 00:00:03'],
        'value1': [1.0, 2.0, 3.0],
        'value2': [4.0, 5.0, 6.0],
        'filename': ['file1', 'file1', 'file1']
    }
    group = pd.DataFrame(data)
    group['time'] = pd.to_datetime(group['time'])
    detector = FallDetector(window_size=100, overlap=50, data_loaders=[])
    result = detector._resample_data(group)

    # Check that resampling has filled the gap correctly
    assert result['time'][0] == pd.Timestamp('2025-05-01 00:00:00')
    assert result['time'][1] == pd.Timestamp('2025-05-01 00:00:00.020')
    assert result['time'][2] == pd.Timestamp('2025-05-01 00:00:00.040')

    assert result['value1'][1] > 1.0 and result['value1'][1] < 2.0
    assert result['value2'][1] > 4.0 and result['value2'][1] < 5.0


def test_resample_data_interpolation():
    # 20ms gap between values
    data = {
        'time': ['2025-05-01 00:00:00.000', '2025-05-01 00:00:00.041'], 
        'start_time': ['2025-05-01 00:00:00', '2025-05-01 00:00:00'],
        'end_time': ['2025-05-01 00:00:01', '2025-05-01 00:00:01'],
        'value1': [1.0, 3.0],  
        'value2': [5.0, 7.0],  
        'filename': ['file1', 'file1']
    }
    group = pd.DataFrame(data)
    group['time'] = pd.to_datetime(group['time'])
    detector = FallDetector(window_size=100, overlap=50, data_loaders=[])
    result = detector._resample_data(group)

    assert result['time'][0] == pd.Timestamp('2025-05-01 00:00:00')
    assert result['time'][1] == pd.Timestamp('2025-05-01 00:00:00.020')
    assert result['time'][2] == pd.Timestamp('2025-05-01 00:00:00.040')
    assert result['value1'][1] > 1.0 and result['value1'][1] < 3.0  
    assert result['value2'][1] > 5.0 and result['value2'][1] < 7.0  


def test_resample_interpolation_with_duplicates():
    # 20ms gap between values
    data = {
        'time': ['2025-05-01 00:00:00.000', '2025-05-01 00:00:00.020', '2025-05-01 00:00:00.020'], 
        'start_time': ['2025-05-01 00:00:00', '2025-05-01 00:00:00', '2025-05-01 00:00:00'],
        'end_time': ['2025-05-01 00:00:01', '2025-05-01 00:00:01', '2025-05-01 00:00:01'],
        'value1': [1.0, 3.0, 3.0],  
        'value2': [5.0, 7.0, 7.0],  
        'filename': ['file1', 'file1', 'file1']
    }
    group = pd.DataFrame(data)
    group['time'] = pd.to_datetime(group['time'])
    detector = FallDetector(window_size=100, overlap=50, data_loaders=[])
    result = detector._resample_data(group)

    assert result['time'][0] == pd.Timestamp('2025-05-01 00:00:00')
    assert result['time'][1] == pd.Timestamp('2025-05-01 00:00:00.020')
    assert len(result['time'].unique()) == 2

def test_resample_bfill_ffill_interpolate():
    data = {
        'time': ['2025-05-01 00:00:00.000', '2025-05-01 00:00:00.020', '2025-05-01 00:00:00.060', '2025-05-01 00:00:00.080'], 
        'start_time': ['2025-05-01 00:00:00', '2025-05-01 00:00:00', '2025-05-01 00:00:00', '2025-05-01 00:00:00'],
        'end_time': ['2025-05-01 00:00:01', '2025-05-01 00:00:01', '2025-05-01 00:00:01', '2025-05-01 00:00:01'],
        'value1': [None, 1.0, 3.0, None], 
        'value2': [None, 5.0, 7.0, None],  
        'filename': ['file1', 'file1', 'file1', 'file1']
    }
    group = pd.DataFrame(data)
    group['time'] = pd.to_datetime(group['time'])
    detector = FallDetector(window_size=100, overlap=50, data_loaders=[])
    result = detector._resample_data(group)

    assert result['value1'][0] == 1.0  
    assert result['value2'][0] == 5.0  
    assert result['value1'][3] == 3.0  
    assert result['value2'][3] == 7.0  

def test_resample_bfill_ffill_interpolate2():
    data = {
        'time': ['2025-05-01 00:00:00', '2025-05-01 00:00:01'],
        'start_time': ['2025-05-01 00:00:00', '2025-05-01 00:00:00'],
        'end_time': ['2025-05-01 00:00:01', '2025-05-01 00:00:02'],
        'value1': [None, 2.0],
        'value2': [3.0, None],
        'filename': ['file1', 'file1']
    }
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    detector = FallDetector(window_size=100, overlap=50, data_loaders=[])
    result = detector._resample_data(df)
    assert result['value1'][0] is not None, "Missing values should be handled correctly"
    assert result['value2'][1] is not None, "Missing values should be handled correctly"


def test_pre_process_integration():
    # Example input data with the required columns and filename
    input_data = pd.DataFrame({
        'time': ['2025-05-01 00:00:00', '2025-05-01 00:00:01', '2025-05-01 00:00:02'] * 3,
        'start_time': ['2025-05-01 00:00:00', '2025-05-01 00:00:01', '2025-05-01 00:00:02'] * 3,
        'end_time': ['2025-05-01 00:00:01', '2025-05-01 00:00:02', '2025-05-01 00:00:03'] * 3,
        'accel_x_list': [1.0, 1.1, 1.2] * 3,
        'accel_y_list': [0.0, 0.1, 0.2] * 3,
        'accel_z_list': [0.5, 0.6, 0.7] * 3,
        'gyro_x_list': [0.0, -0.1, -0.2] * 3,
        'gyro_y_list': [0.5, 0.4, 0.3] * 3,
        'gyro_z_list': [1.0, 1.1, 1.2] * 3,
        'filename': ['UMA.csv'] * 3 + ['1_6_1.csv'] * 3 + ['U01_R01.csv'] * 3,
    })

    # Ensure the 'time' column is a datetime type before processing
    input_data['time'] = pd.to_datetime(input_data['time'])

    detector = FallDetector(window_size=100, overlap=50, data_loaders=[])
    processed_data = detector.pre_process(input_data)

    assert not processed_data.empty, "Processed data should not be empty"

    required_columns = ['accel_x_list', 'accel_y_list', 'accel_z_list', 
                        'gyro_x_list', 'gyro_y_list', 'gyro_z_list']
    
    for col in required_columns:
        assert col in processed_data.columns, f"Column {col} is missing from processed data"
        assert pd.api.types.is_numeric_dtype(processed_data[col]), f"Column {col} should be numeric"


    assert len(processed_data) != len(input_data), "Resampling should change the data length"
    assert processed_data['accel_x_list'].iloc[0] != input_data['accel_x_list'].iloc[0], "Gaussian filtering should have altered the values"
    assert pd.to_datetime(processed_data['time']).iloc[0] == pd.to_datetime(input_data['time']).iloc[0], "Time should be parsed correctly"

def test_metadata_inclusion():
    data = {
        'gyro_x_list': np.random.rand(100),
        'gyro_y_list': np.random.rand(100),
        'gyro_z_list': np.random.rand(100),
        'accel_x_list': np.random.rand(100),
        'accel_y_list': np.random.rand(100),
        'accel_z_list': np.random.rand(100),
        'time': pd.date_range(start='2025-01-01', periods=100, freq='20ms'),
        'filename': ['test_file'] * 100,
        'start_time': [pd.to_datetime('2025-01-01 00:00:00')] * 100,
        'end_time': [pd.to_datetime('2025-01-01 00:01:40')] * 100
    }

    df = pd.DataFrame(data)
    window_size = 25
    step_size = 10

    result_with_metadata = process_window(df, window_size, step_size, include_metadata=True)
    result_without_metadata = process_window(df, window_size, step_size, include_metadata=False)

    # Check if metadata columns exist when include_metadata is True
    assert 'is_fall' in result_with_metadata.columns, "Expected 'is_fall' column"
    assert 'filename' in result_with_metadata.columns, "Expected 'filename' column"

    # Check that metadata columns are absent when include_metadata is False
    assert 'is_fall' not in result_without_metadata.columns, "Expected no 'is_fall' column"
    assert 'filename' not in result_without_metadata.columns, "Expected no 'filename' column"

def test_edge_cases():
    # Test with a single window
    data = {
        'gyro_x_list': np.random.rand(25),
        'gyro_y_list': np.random.rand(25),
        'gyro_z_list': np.random.rand(25),
        'accel_x_list': np.random.rand(25),
        'accel_y_list': np.random.rand(25),
        'accel_z_list': np.random.rand(25),
        'time': pd.date_range(start='2025-01-01', periods=25, freq='20ms'),
        'filename': ['test_file'] * 25,
        'start_time': [pd.to_datetime('2025-01-01 00:00:00')] * 25,
        'end_time': [pd.to_datetime('2025-01-01 00:00:20')] * 25
    }
    df_single_window = pd.DataFrame(data)
    result_single_window = process_window(df_single_window, window_size=25, step_size=25)

    assert len(result_single_window) == 1, "Expected 1 window for a single input window."
    df_empty = pd.DataFrame(columns=['gyro_x_list', 'gyro_y_list', 'gyro_z_list', 'accel_x_list', 'accel_y_list', 'accel_z_list', 'time', 'filename', 'start_time', 'end_time'])
    result_empty = process_window(df_empty, window_size=26, step_size=25)

    assert len(result_empty) == 0, "Expected 0 windows for an empty DataFrame."


def test_process_window():
    # Create a mock dataframe with columns matching the expected data
    data = {
        'gyro_x_list': np.random.rand(100),
        'gyro_y_list': np.random.rand(100),
        'gyro_z_list': np.random.rand(100),
        'accel_x_list': np.random.rand(100),
        'accel_y_list': np.random.rand(100),
        'accel_z_list': np.random.rand(100),
        'time': pd.date_range(start='2025-01-01', periods=100, freq='20ms'),
        'filename': ['test_file'] * 100,
        'start_time': [pd.to_datetime('2025-01-01 00:00:00')] * 100,
        'end_time': [pd.to_datetime('2025-01-01 00:01:40')] * 100
    }

    df = pd.DataFrame(data)
    window_size = 25
    overlap = 5
    step_size = window_size - overlap
    result = process_window(df, window_size, step_size, include_metadata=True)

    print(result.columns.to_list())
    assert 'accel_magnitude_mean' in result.columns, "Expected 'accel_magnitude_mean' column"
    assert 'gyro_magnitude_mean' in result.columns, "Expected 'gyro_magnitude_mean' column"
    assert 'is_fall' in result.columns, "Expected 'is_fall' column"

    expected_num_windows = (len(df) - window_size) // step_size
    assert len(result) == expected_num_windows, f"Expected {expected_num_windows} windows, but got {len(result)}"
