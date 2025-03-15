from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import Parallel, delayed
from FallDetector import FallDetector
import pandas as pd
import numpy as np
import os
import datetime
import pytz
class ModelTester:
    def __init__(self, configs, datafolder=os.path.join(os.getcwd(), '50Hz'), results_file="results.csv", n_features=50, n_kfolds=5, n_components=0.90):
        """Initialize the tester with multiple configurations."""
        self.configs = configs  # List of configurations (window_size, overlap)
        self.datafolder = datafolder
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
            fall_detector = FallDetector(window_size, overlap, self.datafolder)
            features_file = fall_detector.get_file_path()

            if os.path.exists(features_file):
                print(f"Loading features from {features_file}")
                features_df = pd.read_csv(features_file)
            else:
                print(f"Features file not found. Generating features...")
                df = fall_detector.load_data()
                df = fall_detector.pre_process(df)
                features_df = fall_detector.extract_features(df)

            if features_df.isnull().values.any():
                features_df.dropna(inplace=True)
            
            self.metrics.append(self.evaluate_model(features_df, window_size, overlap))

        final_results = pd.DataFrame(self.metrics)

        # Save to CSV
        os.makedirs('results', exist_ok=True)
        if os.path.exists(self.results_file):
            final_results.to_csv(self.results_file, mode='a', header=False, index=False)
        else:
            final_results.to_csv(self.results_file, index=False)

    def evaluate_model(self, df, window_size, overlap):
        print(f"Evaluating model {window_size, overlap}")
        print(df['is_fall'].value_counts())
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

            # Apply SMOTE on PCA-transformed features
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
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

        # Parallel execution for each fold
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
        
    def visualize_smote(self, features_file_path): 

        #Load data
        df = pd.read_csv(features_file_path)
        X = df.drop(columns=['is_fall', 'filename', 'start_time', 'end_time'])
        y = df['is_fall'].reset_index(drop=True)

        

        # Create imbalanced data
        X, y = make_classification(n_classes=2, weights=[0.9, 0.1], n_features=10, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA to reduce to 2D for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Apply PCA again to visualize synthetic vs. original
        X_resampled_pca = pca.transform(X_resampled)

        # Plot original vs. synthetic data
        plt.figure(figsize=(8,6))
        plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label="Majority Class (Original)", alpha=0.5)
        plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label="Minority Class (Original)", color='red', alpha=0.5)
        plt.scatter(X_resampled_pca[len(y):, 0], X_resampled_pca[len(y):, 1], label="Synthetic Minority (SMOTE)", color='green', alpha=0.5)

        plt.legend()
        plt.xlabel("PCA Feature 1")
        plt.ylabel("PCA Feature 2")
        plt.title("SMOTE: Original vs. Synthetic Data Points")
        plt.show()

# Example usage
tester = ModelTester(configs=[(50, 40)
                               ], n_kfolds=5)
tester.run_tests()
results = tester.get_results()
print(results)


ModelTester(configs=[(50, 40)], n_kfolds=5).visualize_smote(os.path.join(os.getcwd(), 'features', 'features_w50_o40.csv'))