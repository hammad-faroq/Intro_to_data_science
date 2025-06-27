import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# CustomScaler for training (not used during prediction, but useful to include)
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# Final absenteeism model class
class absenteeism_model():
    def __init__(self, model_file='model', scaler_file='scaler'):
        with open(model_file, 'rb') as mf, open(scaler_file, 'rb') as sf:
            self.reg = pickle.load(mf)
            self.scaler = pickle.load(sf)
            self.data = None

    def load_and_clean_data(self, data_file):
        df = pd.read_csv(data_file, delimiter=',')

        # Store raw for final output
        self.df_with_predictions = df.copy()

        # Drop unnecessary columns if present
        for col in ['ID', 'Pet', 'Absenteeism Time in Hours']:
            if col in df.columns:
                df = df.drop(col, axis=1)

        # Add dummy column for pipeline compatibility
        df['Absenteeism Time in Hours'] = np.nan

        # Create dummy variables for Reason for Absence
        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
        reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
        reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)

        df = df.drop(['Reason for Absence'], axis=1)
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)

        # Rename columns safely
        column_names = [
            'Date', 'Transportation Expense', 'Distance to Work', 'Age',
            'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
            'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4'
        ]

        if df.shape[1] == len(column_names):
            df.columns = column_names
        else:
            raise ValueError(f"❌ Column length mismatch: got {df.shape[1]} columns, expected {len(column_names)}. Check your CSV or preprocessing steps.")

        # Reorder columns
        df = df[[
            'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date',
            'Transportation Expense', 'Distance to Work', 'Age',
            'Daily Work Load Average', 'Body Mass Index', 'Education',
            'Children', 'Absenteeism Time in Hours'
        ]]

        # Convert and extract date features
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df['Month Value'] = df['Date'].dt.month
        df['Day of the Week'] = df['Date'].dt.weekday
        df = df.drop(['Date'], axis=1)

        # Final ordering
        final_columns = [
            'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value',
            'Day of the Week', 'Transportation Expense', 'Distance to Work',
            'Age', 'Daily Work Load Average', 'Body Mass Index',
            'Education', 'Children', 'Absenteeism Time in Hours'
        ]
        df = df[final_columns]

        # Fix education and missing values
        df['Education'] = df['Education'].map({1: 0, 2: 1, 3: 1, 4: 1})
        df = df.fillna(0)

        # Drop columns not used during model training
        df = df.drop(['Absenteeism Time in Hours', 'Day of the Week',
                      'Daily Work Load Average', 'Distance to Work'], axis=1)

        # Store cleaned version
        self.preprocessed_data = df.copy()

        # Match training columns exactly (no surprises like 'Pet')
        expected_columns = self.scaler.feature_names_in_
        df = df[[col for col in df.columns if col in expected_columns]]

        # Scale and store the final input
        self.data = self.scaler.transform(df)

    def predicted_probability(self):
        if self.data is not None:
            return self.reg.predict_proba(self.data)[:, 1]

    def predicted_output_category(self):
        if self.data is not None:
            return self.reg.predict(self.data)

    def predicted_outputs(self):
        if self.data is not None:
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:, 1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data
