import pandas as pd
import numpy as np
import os
import datetime
import pytz
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from FallDetector import FallDetector
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
class ModelTester:
    def __init__(self, configs, datafolder):
        """Initialize the tester with multiple configurations."""
        self.configs = configs  # List of configurations (window_size, overlap)
        self.datafolder = datafolder
        self.results = []
    
    def run_tests(self):
        """Runs fall detection with different configurations and evaluates models."""
        for config in self.configs:
            window_size, overlap = config
            print(f"Testing configuration: window_size={window_size}, overlap={overlap}")
            
            fall_detector = FallDetector(window_size, overlap, self.datafolder)
            features_file = fall_detector.get_file_path()
            # Check if we already have the features for this configuration
            if os.path.exists(features_file):
                print(f"Loading features from {features_file}")
                features_df = pd.read_csv(features_file)
            else:
                df = fall_detector.load_data()
                df = fall_detector.pre_process(df)
                features_df = fall_detector.extract_features(df)
            
            #Check for NaN values
            if features_df.isnull().values.any():
                features_df.dropna(inplace=True)


            acc, report, conf_matrix = self.evaluate_model(features_df)
            self.results.append({
                'window_size': window_size,
                'overlap': overlap,
                'accuracy': acc,
                'report': report,
                'conf_matrix': conf_matrix
            })

            results_df = pd.DataFrame(self.results)
            # Save CSV to results folder
            os.makedirs('results', exist_ok=True)
            current_time = datetime.datetime.now(pytz.timezone('Europe/Copenhagen'))
            formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
            filename = f'results_{formatted_time}.csv'
            results_df.to_csv(os.path.join('results', filename), index=False)

            # Show the evaluation report
            print(report)

    def evaluate_model(self, df):
        X = df.drop(columns=['is_fall', 'filename', 'time_start', 'time_end'])
        y = df['is_fall']
        
        # Apply feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply SMOTE to balance the classes
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_scaled, y = smote.fit_resample(X_scaled, y)

        strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_accuracies = []
        all_conf_matrices = []

        # Cross-validation
        for train_index, test_index in strat_kfold.split(X_scaled, y):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Initialize the RandomForestClassifier with class_weight='balanced'
            model = RandomForestClassifier(class_weight='balanced', random_state=42)

            # Initialize RFE for feature selection inside cross-validation loop
            selector = RFE(estimator=model, n_features_to_select=20)  # Reduce features to 10
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)

            # Fit model to selected features
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)

            acc = np.mean(y_pred == y_test)
            fold_accuracies.append(acc)

            conf_matrix = confusion_matrix(y_test, y_pred)
            all_conf_matrices.append(conf_matrix)

            # Print the features selected by RFE and their ranking
            print(f"Selected features for this fold: {X.columns[selector.support_]}")
            print(f"Feature rankings: {dict(zip(X.columns, selector.ranking_))}")

        avg_accuracy = np.mean(fold_accuracies)
        avg_conf_matrix = np.mean(np.array(all_conf_matrices), axis=0)
        avg_conf_matrix_rounded = np.round(avg_conf_matrix).astype(int)

        # Fit the model to the entire dataset for final evaluation
        selector = RFE(estimator=model, n_features_to_select=20)
        X_selected = selector.fit_transform(X_scaled, y)
        model.fit(X_selected, y)
        y_pred = model.predict(X_selected)
        report = classification_report(y, y_pred)
        
        return avg_accuracy, report, avg_conf_matrix_rounded

    def get_results(self):
        """Returns the results of the tests."""
        return pd.DataFrame(self.results).sort_values(by='accuracy', ascending=False)


# Example usage
tester = ModelTester(configs=[(100, 60)], datafolder=os.path.join(os.getcwd(), '50Hz'))
tester.run_tests()
results = tester.get_results()
print(results)