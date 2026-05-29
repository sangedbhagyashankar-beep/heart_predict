import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib
import numpy as np

def preprocess_data(df):
    """
    Performs data cleaning and preprocessing steps.
    - Handles missing values (although none exist in this dataset, it's a good practice).
    - Removes duplicate rows.
    """
    print("Starting data preprocessing...")
    # Check for and handle missing values
    if df.isnull().sum().any():
        print("Warning: Missing values detected. Filling with median for simplicity.")
        df = df.fillna(df.median())
    
    # Remove duplicate rows
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate rows.")
    
    print("Data preprocessing complete.")
    return df

# Load dataset
df = pd.read_csv('heart.csv')

# --- NEW STEP: Call the preprocessing function ---
df = preprocess_data(df)

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training with CalibratedClassifierCV for better probability estimates
rf = RandomForestClassifier(random_state=42)
clf = CalibratedClassifierCV(rf)  # Calibrated for better probability estimates
clf.fit(X_train, y_train)

# Save trained model
joblib.dump(clf, 'heart_model.pkl')

# Access feature importances correctly from the base estimator of CalibratedClassifierCV
# The calibrated_classifiers_ attribute is a list, and each element has an .estimator
importances = clf.calibrated_classifiers_[0].estimator.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances.to_csv('feature_importances.csv', index=False)

print("Model trained and saved as heart_model.pkl")
print("Feature importances saved as feature_importances.csv")