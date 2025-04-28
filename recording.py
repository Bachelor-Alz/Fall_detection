import os 
import numpy as np
import pandas as pd
import httpx

def has_five_consecutive_ones(arr):
    count = 0
    for num in arr:
        if num == 1:
            count += 1
            if count >= 4:
                return True
        else:
            count = 0
    return False

def send_request(df: pd.DataFrame, folder_name: str, file_name: str = None):
    url = "http://localhost:9999/predict"
    headers = {"Content-Type": "application/json"}
    df.dropna(inplace=True)
    
    payload = {
        "mac": "example",
        "readings": df.to_dict(orient="records"),
    }

    response = httpx.post(url, json=payload, headers=headers)
    try:
        response_data : list[np.number] = response.json().get('predictions', [])
        
    except ValueError:
        print(f"Error parsing response for {folder_name}: {response.text}")
        return
    if not response_data:
        print(f"No data received for {folder_name}")
        return
    is_fall = has_five_consecutive_ones(response_data)
    results.loc[len(results)] = [folder_name, is_fall, file_name]

def get_recording_results():
    folder_path = os.path.join(os.getcwd(), folder)
    for activity_folder in os.listdir(folder_path):
        activity_folder_path = os.path.join(folder_path, activity_folder)
        if os.path.isdir(activity_folder_path):
            for file in os.listdir(activity_folder_path):
                file_path = os.path.join(activity_folder_path, file)
                if file.endswith('.csv'):
                    df = pd.read_csv(file_path, header=None, sep=',', skiprows=1, names=cols)
                    df = df[:-1]
                    send_request(df, activity_folder, file)

folder = 'recordings'
cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
results = pd.DataFrame(columns=['folder_name', 'is_fall', 'file_name'])
get_recording_results()

activity_table = pd.DataFrame(columns=['activity', 'is_fall_count'])

for activity in results['folder_name'].unique():
    is_fall_count = results[results['folder_name'] == activity]['is_fall'].sum()
    activity_table.loc[len(activity_table)] = [activity, is_fall_count]

activity_table.to_csv('activity_results.csv', index=False)
results.sort_values(by=['folder_name', 'file_name'], inplace=True)
results.to_csv('recording_results.csv', index=False)