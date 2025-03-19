import os 
import pandas as pd
import numpy as np

class LoadUMA:
    def __init__(self):
        self.datafolder = os.path.join(os.getcwd(), 'UMAFall')
        self.fall_timestamps = pd.read_csv(os.path.join(os.getcwd(), 'UMA_fall_timestamps.csv'))

    
    def load_data(self):
        """Load data from UMAFall dataset."""

        #Check which files are not in the fall_timestamps filename column
        files = os.listdir(self.datafolder)
        files = [f for f in files if f.endswith('.csv')]
        files = [f for f in files if f not in self.fall_timestamps['filename'].values]






#Test the class
load_uma = LoadUMA()
load_uma.load_data()