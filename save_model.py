import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("Training and saving the XGBoost model for the Streamlit app...")

# Load the data
train_data = pd.read_csv('train.csv')

# Function to preprocess data
def preprocess_data(df, is_training=True):
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Drop Loan_ID as it's just an identifier
    if 'Loan_ID' in data.columns:
        data = data.drop('Loan_ID', axis=1)
    
    # Handling missing values for categorical features
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 
                            'Self_Employed', 'Property_Area']
    
    for feature in categorical_features:
        data[feature] = data[feature].fillna(data[feature].mode()[0])
    
    # Handling missing values for numerical features
    numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    
    for feature in numerical_features:
        data[feature] = data[feature].fillna(data[feature].median())
    
    # Fill Credit_History with most common value (mode)
    if 'Credit_History' in data.columns:
        data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])
    
    # Feature Engineering
    # Total Income
    data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    
    # Income to Loan Amount Ratio
    data['Income_Loan_Ratio'] = data['TotalIncome'] / (data['LoanAmount'] + 1)
    
    # EMI Calculation (approximate)
    data['EMI'] = data['LoanAmount'] * 1000 / (data['Loan_Amount_Term'] + 1)
    
    # EMI to Income Ratio
    data['EMI_Income_Ratio'] = data['EMI'] / (data['TotalIncome'] + 1)
    
    # Log transformation for skewed numerical features
    data['ApplicantIncome'] = np.log1p(data['ApplicantIncome'])
    data['CoapplicantIncome'] = np.log1p(data['CoapplicantIncome'])
    data['LoanAmount'] = np.log1p(data['LoanAmount'])
    data['TotalIncome'] = np.log1p(data['TotalIncome'])
    
    # One-hot encoding for categorical features
    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    
    # Prepare features and target
    if is_training and 'Loan_Status' in data.columns:
        # Convert target to binary (0/1)
        le_target = LabelEncoder()
        data['Loan_Status'] = le_target.fit_transform(data['Loan_Status'])
        y = data['Loan_Status']
        X = data.drop('Loan_Status', axis=1)
        return X, y
    else:
        return data, None

# Preprocess training data
print("Preprocessing training data...")
X, y = preprocess_data(train_data, is_training=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model with best parameters
# Note: In a real scenario, you would perform hyperparameter tuning first
print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Train the model
xgb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
print("Saving the model to 'loan_prediction_model.joblib'...")
joblib.dump(xgb_model, 'loan_prediction_model.joblib')

# Save feature columns for reference
joblib.dump(X.columns.tolist(), 'model_columns.joblib')

print("Model and columns saved successfully! Now you can run the Streamlit app.")
print("To launch the app, run: 'streamlit run loan_prediction_app.py'") 