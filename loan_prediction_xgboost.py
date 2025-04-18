import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

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
X, y = preprocess_data(train_data, is_training=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
print("\n----- XGBoost Model -----")
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Feature importance
plt.figure(figsize=(12, 8))
xgb.plot_importance(xgb_model, max_num_features=15)
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()

# Hyperparameter tuning with GridSearchCV
print("\n----- Hyperparameter Tuning -----")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best score: {grid_search.best_score_:.4f}')

# Cross-validation with the best model
best_model = grid_search.best_estimator_
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})')

# Preprocess test data
test_processed, _ = preprocess_data(test_data, is_training=False)

# Make sure test data has the same columns as training data
for col in X.columns:
    if col not in test_processed.columns:
        test_processed[col] = 0
        
# Reorder columns to match training data
test_processed = test_processed[X.columns]

# Make predictions on test data
test_predictions = best_model.predict(test_processed)

# Convert predictions to 'Y' and 'N'
test_predictions_labels = ['Y' if pred == 1 else 'N' for pred in test_predictions]

# Create submission dataframe
submission = pd.DataFrame({
    'Loan_ID': test_data['Loan_ID'],
    'Loan_Status': test_predictions_labels
})

# Save the submission file
submission.to_csv('submission_xgboost.csv', index=False)

print("\nXGBoost submission file created successfully.") 