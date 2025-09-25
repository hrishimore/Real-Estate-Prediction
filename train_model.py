# -------------------------------------------------------------------
# REAL ESTATE PRICE PREDICTION - MODEL TRAINING SCRIPT
# -------------------------------------------------------------------
# This script performs the following steps:
# 1. Loads the Ames Housing dataset.
# 2. Performs data cleaning and preprocessing.
# 3. Engineers relevant features for the model.
# 4. Splits the data into training and testing sets.
# 5. Trains a Gradient Boosting Regressor model.
# 6. Evaluates the model's performance.
# 7. Saves the trained model to a file for future use.
# -------------------------------------------------------------------

# 1. Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib # For saving the model

print("Script started: Loading libraries...")

# 2. Load the dataset
# For a real project, you would use the full dataset.
# Here, we use the provided sample data.
try:
    df = pd.read_csv('data/ames_housing_sample.csv')
    print("Data loaded successfully.")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'data/ames_housing_sample.csv' not found.")
    print("Please make sure the data file is in a 'data' subfolder.")
    exit()


# 3. Feature Engineering & Preprocessing
print("Starting feature engineering and preprocessing...")

# Create 'PropertyAge' feature
df['PropertyAge'] = df['YearSold'] - df['YearBuilt']

# Handle missing values simply for this example
# In a full project, this would be more sophisticated
df['GarageCars'] = df['GarageCars'].fillna(df['GarageCars'].median())
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())

# Select features for the model
# These are some of the most impactful features from the dataset
features = [
    'OverallQual', 
    'GrLivArea', 
    'GarageCars', 
    'TotalBsmtSF', 
    'FullBath', 
    'YearBuilt', 
    'PropertyAge',
    'LotFrontage'
]
target = 'SalePrice'

# Ensure all selected features exist and handle potential missing ones
for feature in features:
    if feature not in df.columns:
        print(f"Error: Feature '{feature}' not found in the dataset.")
        exit()
    if df[feature].isnull().any():
        print(f"Warning: Missing values found in '{feature}'. Filling with median.")
        df[feature] = df[feature].fillna(df[feature].median())


X = df[features]
y = df[target]

print("Preprocessing complete.")


# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")


# 5. Train the Gradient Boosting Regressor Model
print("Training the Gradient Boosting model...")
# Model parameters can be tuned for better performance
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)
print("Model training complete.")


# 6. Evaluate the model
print("Evaluating model performance...")
y_pred = gbr.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"Model Performance on Test Set:")
print(f"R-squared (R²): {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")


# 7. Save the trained model to a file
model_filename = 'real_estate_model.joblib'
joblib.dump(gbr, model_filename)
print(f"Model saved successfully as '{model_filename}'")
print("Script finished.")
