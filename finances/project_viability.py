import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import joblib
import os


class ProjectViability:
    def __init__(self, model_path_name='finances\logistic_model', project_data_path_name='finances\projects_data', test_size=0.2, random_state=42):
        self.model_path = f'{model_path_name}.joblib'
        self.model = None
        self.scaler = None
        self.report = None
        self.project_data_csv = f'{project_data_path_name}.csv'
        self.df_projects = None
        self.x = None
        self.y = None
        self.X_scaled = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_size = test_size
        self.random_state = random_state


    def _model_exists(self):
        """
        Check if the model file exists in the current directory.
        :return: True if the model file exists, False otherwise.
        """
        return os.path.exists(self.model_path)
    
    def _load_model(self):
        """
        Load the pre-trained model and scaler from disk.
        :return: None
        """
        try:
            self.model = joblib.load(self.model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file '{self.model_path}' not found.")
        try:
            self.scaler = joblib.load(self.model_path.replace('.joblib', '_scaler.joblib'))
        except FileNotFoundError:
            raise FileNotFoundError(f"Scaler file '{self.model_path.replace('.joblib', '_scaler.joblib')}' not found.")

    def _load_data(self):
        """
        Load the project data from a CSV file.
        :return: None
        """
        try:
            self.df_projects = pd.read_csv(self.project_data_csv)
        except FileNotFoundError:
            raise FileNotFoundError(f"Project data file '{self.project_data_csv}' not found.")
        
    def _prepare_data(self):
        """
        Prepare the data for model training or prediction.
        :return: None
        """
        try:
            self.x = self.df_projects[["investment", "expected_return", "impact_score"]]
            self.y = self.df_projects["viability"]
        except KeyError as e:
            raise KeyError(f"Missing expected columns in the data: {e}")
        
    def _normalize_data(self):
        """
        Normalize the feature data using StandardScaler.
        :return: None
        """
        try:
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(self.x)
        except Exception as e:
            raise Exception(f"Error during data normalization: {e.__context__}")
        
    def _split_train_test(self):
        """
        Split the normalized data into training and testing sets.
        :return: None
        """
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X_scaled,
                self.y,
                test_size=self.test_size,
                random_state=self.random_state
            )
        except Exception as e:
            raise Exception(f"Error during train-test split: {e.__context__}")

    def _train_model(self):
        """
        Create a new logistic regression model and scaler.
        :return: None
        """
        try:
            self.model = LogisticRegression()
            self.model.fit(self.X_train, self.y_train)
        except Exception as e:
            raise Exception(f"Error during model training: {e.__context__}")
        
    def _evaluate_model(self):
        '''
        Evaluate the model using the test set and generate a classification report.
        :return: None
        '''
        try:
            y_pred = self.model.predict(self.X_test)
            self.report = classification_report(self.y_test, y_pred, output_dict=True)
        except Exception as e:
            raise Exception(f"Error during model evaluation: {e.__context__}")

    def _save_model(self):
        '''
        Save the trained model, scaler, and report to disk.
        :return: None
        '''
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.model_path.replace('.joblib', '_scaler.joblib'))
            joblib.dump(self.report, self.model_path.replace('.joblib', '_metrics.joblib'))
        except Exception as e:
            raise Exception(f"Error during model saving: {e.__context__}")
        
    def _set_model(self):
        """
        Set the model by loading it if it exists, or training a new one if it doesn't.
        :return: None
        """
        if self._model_exists():
            self._load_model()
        else:
            self._load_data()
            self._prepare_data()
            self._normalize_data()
            self._split_train_test()
            self._train_model()
            self._evaluate_model()
            self._save_model()

    def predict_viability(self, new_projects):
        """
        Predict the viability of new projects based on the trained model.
        :param new_projects: List of dictionaries containing project data.
        :return: DataFrame with predictions and probabilities, or None if an error occurs.
        """
        self._set_model()
        try:
            # Convert new projects to DataFrame and scale the features
            df_new_projects = pd.DataFrame(new_projects)
            X_new_scaled = self.scaler.transform(df_new_projects)
            predictions = self.model.predict(X_new_scaled)

            # Probability by 1
            probabilities = self.model.predict_proba(X_new_scaled)[:, 1]
            df_new_projects["probability"] = probabilities
            df_new_projects["viability"] = predictions

            return df_new_projects, joblib.load(self.model_path.replace('.joblib', '_metrics.joblib'))
        except Exception:
            return None, joblib.load(self.model_path.replace('.joblib', '_metrics.joblib'))
        

if __name__ == "__main__":
    # Novos projetos
    new_projects = [
        #{"investment": 13000, "expected_return": 69000, "impact_score": 7}
        {"investment": 40000, "expected_return": 60000, "impact_score": 6}
    ]

    viability_checker = ProjectViability()
    predictions, metrics = viability_checker.predict_viability(new_projects)

    if predictions is not None:
        print("\nNovos Projetos e Viabilidade:")
        print(predictions)

    print("\nMétricas do Modelo:")
    print(metrics)