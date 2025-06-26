import pandas as pd
import ast

def check_consecutive_ones(predictions_list, length):
    """
    Checks if a list contains a sequence of 'length' or more consecutive ones.

    Args:
        predictions_list (list): A list of integers (0s and 1s).
        length (int): The minimum required number of consecutive ones.

    Returns:
        bool: True if the sequence is found, False otherwise.
    """
    current_consecutive = 0
    for val in predictions_list:
        if val == 1:
            current_consecutive += 1
        else:
            current_consecutive = 0  # Reset count if a 0 is encountered
        if current_consecutive >= length:
            return True
    return False

# Define the CSV file name
csv_file_name = 'recording_results.csv'

# Load the CSV file
try:
    df = pd.read_csv(csv_file_name)
except FileNotFoundError:
    print(
        f"Error: '{csv_file_name}' not found. "
        "Please make sure the file is in the same directory as the script."
    )
    exit()

# Convert the 'predictions' string column into actual Python lists
# ast.literal_eval is used for safe parsing of string representations of Python
# literals.
df['predictions_list'] = df['predictions'].apply(ast.literal_eval)

# Dictionary to store the counts for each folder
folder_consecutive_counts = {}

# Group the DataFrame by 'folder_name'
grouped_by_folder = df.groupby('folder_name')

# Iterate through each folder group
for folder_name, group_df in grouped_by_folder:
    # Initialize counts for the current folder
    count_3_consecutive = 0
    count_4_consecutive = 0
    count_5_consecutive = 0
    
    # Get the total number of entries for this folder (should be 15)
    total_entries_in_folder = len(group_df)

    # Iterate through each row in the current folder's group
    for _, row in group_df.iterrows():
        predictions = row['predictions_list']

        # Check for 3, 4, and 5 consecutive ones
        # A row with 5 consecutive ones will also satisfy the conditions
        # for 3 and 4.
        if check_consecutive_ones(predictions, 3):
            count_3_consecutive += 1
        if check_consecutive_ones(predictions, 4):
            count_4_consecutive += 1
        if check_consecutive_ones(predictions, 5):
            count_5_consecutive += 1
    
    # Store the results for the current folder
    folder_consecutive_counts[folder_name] = {
        'total': total_entries_in_folder,
        '3s': count_3_consecutive,
        '4s': count_4_consecutive,
        '5s': count_5_consecutive,
    }

# Print the final results in the desired format
print("Consecutive '1's counts per folder:")
for folder, counts in folder_consecutive_counts.items():
    print(
        f"{folder}: "
        f"3 consecutive ones: {counts['3s']}/{counts['total']}, "
        f"4 consecutive ones: {counts['4s']}/{counts['total']}, "
        f"5 consecutive ones: {counts['5s']}/{counts['total']}"
    )