import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
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
    
    # Log transformation for skewed numerical features
    data['ApplicantIncome'] = np.log1p(data['ApplicantIncome'])
    data['CoapplicantIncome'] = np.log1p(data['CoapplicantIncome'])
    data['LoanAmount'] = np.log1p(data['LoanAmount'])
    data['TotalIncome'] = np.log1p(data['TotalIncome'])
    
    # Label encoding for categorical features
    label_encoders = {}
    
    for feature in categorical_features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])
        label_encoders[feature] = le
    
    # Prepare features and target
    if is_training and 'Loan_Status' in data.columns:
        # Convert target to binary (0/1)
        le_target = LabelEncoder()
        data['Loan_Status'] = le_target.fit_transform(data['Loan_Status'])
        y = data['Loan_Status']
        X = data.drop('Loan_Status', axis=1)
        return X, y, label_encoders
    else:
        return data, None, label_encoders

# Preprocess training data
X, y, label_encoders = preprocess_data(train_data, is_training=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')
    print('\nConfusion Matrix:')
    print(cm)
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    return model

# Train and evaluate Logistic Regression model
print("\n----- Logistic Regression Model -----")
lr_model = LogisticRegression(random_state=42)
lr_model = evaluate_model(lr_model, X_train, X_test, y_train, y_test)

# Train and evaluate Random Forest model
print("\n----- Random Forest Model -----")
rf_model = RandomForestClassifier(random_state=42)
rf_model = evaluate_model(rf_model, X_train, X_test, y_train, y_test)

# Feature importance for Random Forest
plt.figure(figsize=(12, 6))
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()

# Cross-validation to check for overfitting
print("\n----- Cross-Validation Results -----")
models = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model
}

for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f'{name} CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})')

# Preprocess test data
test_processed, _, _ = preprocess_data(test_data, is_training=False)

# Make predictions on test data
test_predictions_lr = lr_model.predict(test_processed)
test_predictions_rf = rf_model.predict(test_processed)

# Convert predictions to 'Y' and 'N'
test_predictions_lr_labels = ['Y' if pred == 1 else 'N' for pred in test_predictions_lr]
test_predictions_rf_labels = ['Y' if pred == 1 else 'N' for pred in test_predictions_rf]

# Create submission dataframes
submission_lr = pd.DataFrame({
    'Loan_ID': test_data['Loan_ID'],
    'Loan_Status': test_predictions_lr_labels
})

submission_rf = pd.DataFrame({
    'Loan_ID': test_data['Loan_ID'],
    'Loan_Status': test_predictions_rf_labels
})

# Save the submission files
submission_lr.to_csv('submission_logistic_regression.csv', index=False)
submission_rf.to_csv('submission_random_forest.csv', index=False)

print("\nSubmission files created successfully.") 