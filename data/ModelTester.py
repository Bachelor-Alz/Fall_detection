from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import Parallel, delayed, dump
from FallDetector import FallDetector
from LoadData import UmaFallLoader, UpFallLoader, WedaFallLoader
import pandas as pd
import numpy as np
import os
import datetime
import pytz

class ModelTester:
    def __init__(self, configs, results_file="results.csv", n_features=50, n_kfolds=5, n_components=0.80):
        """Initialize the tester with multiple configurations."""
        self.configs = configs  # List of configurations (window_size, overlap)
        self.results_file = results_file  # Path to store results
        self.n_features = n_features
        self.n_kfolds = n_kfolds
        self.n_components = n_components  # The amount of variance to preserve with PCA
        self.metrics = []

    def run_tests(self):
        """Runs fall detection with different configurations and evaluates models."""
        for config in self.configs:
            window_size, overlap = config
            print(f"Testing configuration: window_size={window_size}, overlap={overlap}")
            uma_loader = UmaFallLoader(os.path.join(os.getcwd(), 'UMAFall'), 'UMA_fall_timestamps.csv')
            weda_loader = WedaFallLoader(os.path.join(os.getcwd(), 'WEDAFall'), 'WEDA_fall_timestamps.csv')
            up_fall_loader = UpFallLoader(os.path.join(os.getcwd(), 'UpFall'), 'UP_fall_timestamps.csv')
            fall_detector = FallDetector(window_size, overlap, [uma_loader, weda_loader, up_fall_loader])
            features_file = fall_detector.get_file_path()

            if os.path.exists(features_file):
                print(f"Loading features from {features_file}")
                features_df = pd.read_csv(features_file)
            else:
                print(f"Features file not found. Generating features...")
                df = fall_detector.load_data()
                df = fall_detector.pre_process(df)
                features_df = fall_detector.extract_features(df)
            
            # Print is_fall distribution 0 and 1
            print(features_df['is_fall'].value_counts(normalize=True))
            features_df.dropna(axis=0, how='any', inplace=True)
            self.metrics.append(self.evaluate_model(features_df, window_size, overlap))

        final_results = pd.DataFrame(self.metrics)
        if os.path.exists(self.results_file):
            final_results.to_csv(self.results_file, mode='a', header=False, index=False)
        else:
            final_results.to_csv(self.results_file, index=False)
        
        self.train_and_save_model(features_df, window_size, overlap)

    def evaluate_model(self, df, window_size, overlap):
        print(f"Evaluating model {window_size, overlap}")
        X = df.drop(columns=['is_fall', 'filename', 'start_time', 'end_time'])
        y = df['is_fall'].reset_index(drop=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=self.n_components)  # n_components = 0.95 means preserving 95% of variance
        X_pca = pca.fit_transform(X_scaled)

        strat_kfold = StratifiedKFold(n_splits=self.n_kfolds, shuffle=True, random_state=42)
        n_jobs = max(1, os.cpu_count() // 2 + 1)  # Adjust n_jobs based on available CPU cores

        # Function to process each fold
        def process_fold(train_index, test_index, fold_num):
            print(f"Running fold {fold_num}/{self.n_kfolds}")
            X_train, X_test = X_pca[train_index], X_pca[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            smote = SMOTE(sampling_strategy='minority', random_state=42, k_neighbors=20)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            
            model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight={0: 1, 1:3})
            model.fit(X_train_smote, y_train_smote)  # Train on balanced dataset
            y_pred = model.predict(X_test)

            # Print feature importances
            pca_importance = np.abs(pca.components_.T @ pca.explained_variance_ratio_)
            feature_importance_pca = pd.DataFrame({'Feature': X.columns, 'PCA_Weighted_Importance': pca_importance})
            feature_importance_pca = feature_importance_pca.sort_values(by='PCA_Weighted_Importance', ascending=False)
            print(feature_importance_pca.to_string())

            # Return a dictionary of metrics for each fold
            return {
                "balanced_acc": balanced_accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "conf_matrix": confusion_matrix(y_test, y_pred)
            }

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_fold)(train_idx, test_idx, fold + 1)
            for fold, (train_idx, test_idx) in enumerate(strat_kfold.split(X_pca, y))
        )

        # Aggregate the results
        avg_metrics = {key: np.mean([res[key] for res in results]) for key in ["balanced_acc", "precision", "recall", "f1"]}
        avg_conf_matrix = np.mean([res["conf_matrix"] for res in results], axis=0).astype(int)
        tn, fp, fn, tp = avg_conf_matrix.ravel()

        # Return a DataFrame with the aggregated results
        return {
            'date': datetime.datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'window_size': window_size,
            'overlap': overlap,
            'n_features': self.n_features,
            'n_kfolds': self.n_kfolds,
            'balanced_accuracy': avg_metrics["balanced_acc"],
            'precision': avg_metrics["precision"],
            'recall': avg_metrics["recall"],
            'f1_score': avg_metrics["f1"],
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        }

    def get_results(self):
        """Returns the results of the tests."""
        if os.path.exists(self.results_file):
            return pd.read_csv(self.results_file).sort_values(by='balanced_accuracy', ascending=False)
        else:
            return self.metrics
        
    def train_and_save_model(self, df : pd.DataFrame, window_size, overlap):
        """Train the model and save the PCA transformation and model to disk."""
        print(f"Training and saving model for {window_size, overlap}")
        X = df.drop(columns=['is_fall', 'filename', 'start_time', 'end_time'])
        y = df['is_fall'].reset_index(drop=True)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=self.n_components)
        X_pca = pca.fit_transform(X_scaled)  # Fit PCA first
        num_components = X_pca.shape[1]  # Get actual number of components
        X_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(num_components)])
        smote = SMOTE(sampling_strategy='minority', random_state=42, k_neighbors=20)
        X_smote, y_smote = smote.fit_resample(X_pca, y)

        model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight={0: 1, 1: 3})
        model.fit(X_smote, y_smote)

        # Save the PCA and model
        model_dir = os.path.join(os.pardir)
        model_filename = "fall_detection_model.joblib"
        pca_filename = "pca_transformation.joblib"
        scaler_filename = "features_scaler.joblib"
        model_filename = os.path.join(model_dir, model_filename)
        pca_filename = os.path.join(model_dir, pca_filename)
        scaler_filename = os.path.join(model_dir, scaler_filename)

        dump(model, model_filename)
        dump(pca, pca_filename)
        dump(scaler, scaler_filename)
        
        print(f"Model and PCA saved as {model_filename} and {pca_filename}")
        
    def visualize_smote(self, features_file_path): 
        # Load data
        df = pd.read_csv(features_file_path)
        X = df.drop(columns=['is_fall', 'filename', 'start_time', 'end_time'])
        y = df['is_fall'].reset_index(drop=True)

        # Create imbalanced data
        X, y = make_classification(n_classes=2, weights=[0.9, 0.1], n_features=10, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA to reduce to 3D for visualization
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        # Apply SMOTE
        smote = SMOTE(sampling_strategy='minority', random_state=42, k_neighbors=3)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Apply PCA again to visualize synthetic vs. original
        X_resampled_pca = pca.transform(X_resampled)

        # Plot original vs. synthetic data in 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], X_pca[y == 0, 2], label="Majority Class (Original)", alpha=0.5)
        ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], X_pca[y == 1, 2], label="Minority Class (Original)", color='red', alpha=0.5)
        ax.scatter(X_resampled_pca[len(y):, 0], X_resampled_pca[len(y):, 1], X_resampled_pca[len(y):, 2], label="Synthetic Minority (SMOTE)", color='green', alpha=0.5)

        ax.set_xlabel("PCA Feature 1")
        ax.set_ylabel("PCA Feature 2")
        ax.set_zlabel("PCA Feature 3")
        ax.set_title("SMOTE: Original vs. Synthetic Data Points (3D)")
        ax.legend()
        plt.show()

if __name__ == '__main__':
    # Example usage
        

    tester = ModelTester(configs=[
    (80, 75)
                                ], n_kfolds=5)
    tester.run_tests()
    results = tester.get_results()
    print(results) 
    


    #ModelTester(configs=[(50, 41)], n_kfolds=5).visualize_smote(os.path.join(os.getcwd(), 'features', 'features_w45_o40.csv'))


