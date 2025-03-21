import os
import pandas as pd

# Define folder path
folder = os.path.join(os.getcwd(), 'UpFall', 'Fall')

# Initialize an empty list to store file data
data = []

# Iterate through files in the folder
for file in os.listdir(folder):
    data.append({"filename": file, "start_time": 0, "end_time": 0})

# Convert list to DataFrame
df = pd.DataFrame(data, columns=['filename', 'start_time', 'end_time'])

# Save DataFrame to CSV
df.to_csv('UP_fall_timestamps.csv', index=False)

print("CSV saved successfully.")
