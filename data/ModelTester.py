import pandas as pd
import numpy as np
import os
import datetime
import pytz
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from FallDetector import FallDetector
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed

class ModelTester:
    def __init__(self, configs, datafolder=os.path.join(os.getcwd(), '50Hz'), results_file="results.csv", n_features=20, n_kfolds=5):
        """Initialize the tester with multiple configurations."""
        self.configs = configs  # List of configurations (window_size, overlap)
        self.datafolder = datafolder
        self.results_file = results_file  # Path to store results
        self.results = []
        self.n_features = n_features
        self.n_kfolds = n_kfolds

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
                df = fall_detector.load_data()
                df = fall_detector.pre_process(df)
                features_df = fall_detector.extract_features(df)
                ##TODO REMOVE RETURN
                return

            if features_df.isnull().values.any():
                features_df.dropna(inplace=True)

            metrics_df = self.evaluate_model(features_df, window_size, overlap)
            self.results.append(metrics_df)


        final_results_df = pd.DataFrame(self.results)
        os.makedirs('results', exist_ok=True)
        if os.path.exists(self.results_file):
            final_results_df.to_csv(self.results_file, mode='a', header=False, index=False)
        else:
            final_results_df.to_csv(self.results_file, index=False)

    def evaluate_model(self, df, window_size, overlap):
        print(f"Evaluating model {window_size, overlap}")

        X = df.drop(columns=['is_fall', 'filename', 'time_start', 'time_end'])
        y = df['is_fall'].reset_index(drop=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        strat_kfold = StratifiedKFold(n_splits=self.n_kfolds, shuffle=True, random_state=42)
        n_jobs = max(1, os.cpu_count() // 2)  

        def process_fold(train_index, test_index, fold_num):
            print(f"Running fold {fold_num}/{self.n_kfolds}")
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Feature selection before SMOTE
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            selector = RFE(estimator=model, n_features_to_select=self.n_features)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)

            # Apply SMOTE on selected features
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_selected, y_train)
            
            model.fit(X_train_smote, y_train_smote)  # Train on balanced dataset
            y_pred = model.predict(X_test_selected)

            return {
                "balanced_acc": balanced_accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "conf_matrix": confusion_matrix(y_test, y_pred)
            }

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_fold)(train_idx, test_idx, fold + 1)
            for fold, (train_idx, test_idx) in enumerate(strat_kfold.split(X_scaled, y))
        )

        avg_metrics = {key: np.mean([res[key] for res in results]) for key in ["balanced_acc", "precision", "recall", "f1"]}
        avg_conf_matrix = np.mean([res["conf_matrix"] for res in results], axis=0).astype(int)
        tn, fp, fn, tp = avg_conf_matrix.ravel()

        return pd.DataFrame([{
                'date': datetime.datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'window_size': window_size,
                'overlap': overlap,
                'balanced_accuracy': avg_metrics["balanced_acc"],
                'precision': avg_metrics["precision"],
                'recall': avg_metrics["recall"],
                'f1_score': avg_metrics["f1"],
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'true_positives': tp
            }])


    def get_results(self):
        """Returns the results of the tests."""
        if os.path.exists(self.results_file):
            return pd.read_csv(self.results_file).sort_values(by='balanced_accuracy', ascending=False)
        else:
            return pd.DataFrame(columns=['date', 'window_size', 'overlap', 'balanced_accuracy', 'precision', 'recall', 'f1_score',
                                         'true_negatives', 'false_positives', 'false_negatives', 'true_positives'])


# Example usage
tester = ModelTester(configs=[(150, 60)])
tester.run_tests()
results = tester.get_results()
print(results)
