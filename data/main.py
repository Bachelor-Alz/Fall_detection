import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np

class FallDetection:
    def __init__(self, datafolder, window_size=1.75, overlap=0.5):
        self.datafolder = datafolder
        self.window_size = window_size
        self.overlap = overlap
        self.features_file = self._get_features_filename()
        self.all_features = []

    def _load_and_clean_data(self, accel_file, gyro_file):
        """Load accelerometer and gyroscope data, clean and merge them."""
        accel_data = pd.read_csv(accel_file)
        gyro_data = pd.read_csv(gyro_file)

        accel_data = accel_data.rename(columns={'accel_time_list': 'time'})
        gyro_data = gyro_data.rename(columns={'gyro_time_list': 'time'})
        merged_data = pd.merge(accel_data, gyro_data, on='time', how='outer')
        merged_data = merged_data.sort_values(by='time').interpolate()

        return merged_data

    def _preprocess_data(self, df):
        """Scale accelerometer and gyroscope data."""
        scaler = StandardScaler()
        accel_columns = ['accel_x_list', 'accel_y_list', 'accel_z_list']
        gyro_columns = ['gyro_x_list', 'gyro_y_list', 'gyro_z_list']
        
        df[accel_columns] = scaler.fit_transform(df[accel_columns])
        df[gyro_columns] = scaler.fit_transform(df[gyro_columns])
        
        return df

    def _compute_statistics(self, window_df, axis):
        """Compute basic statistics and features for each axis."""
        stats = {
            f'mean_{axis}': window_df[axis].mean(),
            f'std_{axis}': window_df[axis].std(),
            f'min_{axis}': window_df[axis].min(),
            f'max_{axis}': window_df[axis].max(),
            f'median_{axis}': window_df[axis].median(),
            f'iqr_{axis}': window_df[axis].quantile(0.75) - window_df[axis].quantile(0.25),
            f'rms_{axis}': np.sqrt((window_df[axis] ** 2).mean()),
            f'energy_{axis}': (window_df[axis] ** 2).sum()
        }
        return stats

    def _extract_features(self, df, activity_label):
        """Extract features from the windowed data."""
        window_features = []
        start_time = df['time'].min()
        end_time = df['time'].max()

        while start_time + self.window_size <= end_time:
            window_df = df[(df['time'] >= start_time) & (df['time'] < start_time + self.window_size)]

            if not window_df.empty:
                features = {'start_time': start_time, 'end_time': start_time + self.window_size}

                for axis in ['accel_x_list', 'accel_y_list', 'accel_z_list', 'gyro_x_list', 'gyro_y_list', 'gyro_z_list']:
                    if axis in df.columns:
                        features.update(self._compute_statistics(window_df, axis))

                # Signal Magnitude Area (SMA)
                features['SMA_accel'] = window_df[['accel_x_list', 'accel_y_list', 'accel_z_list']].abs().sum(axis=1).mean()
                features['SMA_gyro'] = window_df[['gyro_x_list', 'gyro_y_list', 'gyro_z_list']].abs().sum(axis=1).mean()
                features['activity'] = 1 if 'F' in activity_label else 0
                window_features.append(features)

            start_time += self.window_size * (1 - self.overlap)
        
        return window_features

    def _get_features_filename(self):
        """Return the filename for saving features based on window size and overlap."""
        return f"features_window{self.window_size}_overlap{self.overlap}.csv"

    def process_data(self):
        """Load or generate feature data."""
        if os.path.exists(self.features_file):
            features_df = pd.read_csv(self.features_file)
        else:
            print(f"Generating features: window_size={self.window_size}, overlap={self.overlap}...")
            for activity_folder in os.listdir(self.datafolder):
                activity_path = os.path.join(self.datafolder, activity_folder)

                if os.path.isdir(activity_path):
                    accel_file, gyro_file = None, None
                    for file in os.listdir(activity_path):
                        if file.endswith("_accel.csv"):
                            accel_file = os.path.join(activity_path, file)
                        elif file.endswith("_gyro.csv"):
                            gyro_file = os.path.join(activity_path, file)

                    if accel_file and gyro_file:
                        merged_data = self._load_and_clean_data(accel_file, gyro_file)
                        preprocessed_data = self._preprocess_data(merged_data)
                        extracted_windows = self._extract_features(preprocessed_data, activity_folder)
                        self.all_features.extend(extracted_windows)

            features_df = pd.DataFrame(self.all_features)
            features_df.to_csv(self.features_file, index=False)
            print(f"Features saved to {self.features_file}.")

        return features_df

    def _train_classifier(self, features_df):
        """Train Random Forest classifier."""
        X = features_df.drop(columns=['start_time', 'end_time', 'activity'])
        y = features_df['activity']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        return model, X, y

    def run_permutations(self, window_sizes, overlaps, n_splits=5):
        """Run permutations for window sizes and overlaps to find the best configuration."""
        results = []

        for window_size, overlap in itertools.product(window_sizes, overlaps):
            self.window_size = window_size
            self.overlap = overlap
            self.features_file = self._get_features_filename()

            features_df = self.process_data()
            model, X, y = self._train_classifier(features_df)

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_accuracies = []
            fold_confusion_matrices = []

            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                fold_accuracies.append((y_pred == y_test).mean())
                cm = confusion_matrix(y_test, y_pred)
                fold_confusion_matrices.append(cm)

            avg_accuracy = np.mean(fold_accuracies)
            avg_confusion_matrix = np.mean(fold_confusion_matrices, axis=0)

            results.append((window_size, overlap, avg_accuracy, avg_confusion_matrix))

        sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
        return sorted_results


# Example usage
datafolder = os.path.join(os.path.dirname(__file__), '40Hz')
window_sizes = [2.5, 4]  
overlaps = [0.6, 0.7]

fall_detection = FallDetection(datafolder)
sorted_results = fall_detection.run_permutations(window_sizes, overlaps)

print("Configurations sorted from best to worst (based on accuracy):")
for result in sorted_results:
    window_size, overlap, avg_accuracy, _ = result
    print(f"Window size = {window_size}s, Overlap = {overlap}, Accuracy = {avg_accuracy:.4f}")

best_result = sorted_results[0]
best_conf_matrix = best_result[3]
np.savetxt("best_confusion_matrix.csv", best_conf_matrix, delimiter=",")
print("Best confusion matrix saved to 'best_confusion_matrix.csv'.")
