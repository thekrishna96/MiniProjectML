import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import os

# Set page title and configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("Loan Approval Prediction System")
st.markdown("Enter the details of the loan applicant and get prediction on loan approval.")

# Function to preprocess input data similar to training data
def preprocess_input(data):
    # Convert input to DataFrame
    input_df = pd.DataFrame(data, index=[0])
    
    # Feature Engineering
    # Total Income
    input_df['TotalIncome'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
    
    # Income to Loan Amount Ratio
    input_df['Income_Loan_Ratio'] = input_df['TotalIncome'] / (input_df['LoanAmount'] + 1)
    
    # EMI Calculation (approximate)
    input_df['EMI'] = input_df['LoanAmount'] * 1000 / (input_df['Loan_Amount_Term'] + 1)
    
    # EMI to Income Ratio
    input_df['EMI_Income_Ratio'] = input_df['EMI'] / (input_df['TotalIncome'] + 1)
    
    # Apply log transformation
    input_df['ApplicantIncome'] = np.log1p(input_df['ApplicantIncome'])
    input_df['CoapplicantIncome'] = np.log1p(input_df['CoapplicantIncome'])
    input_df['LoanAmount'] = np.log1p(input_df['LoanAmount'])
    input_df['TotalIncome'] = np.log1p(input_df['TotalIncome'])
    
    # One-hot encoding for categorical features
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 
                           'Self_Employed', 'Property_Area']
    
    input_encoded = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)
    
    return input_encoded

# Function to load model and columns
def load_model_and_columns():
    model_path = "loan_prediction_model.joblib"
    columns_path = "model_columns.joblib"
    
    if os.path.exists(model_path) and os.path.exists(columns_path):
        model = joblib.load(model_path)
        model_columns = joblib.load(columns_path)
        return model, model_columns
    elif os.path.exists(model_path) and not os.path.exists(columns_path):
        model = joblib.load(model_path)
        st.warning("Model columns file not found. Feature matching may not be accurate.")
        return model, None
    else:
        st.error(f"Model file not found: {model_path}")
        st.info("Please run 'python save_model.py' first to train and save the model.")
        return None, None

# Sidebar for inputs
st.sidebar.header("Applicant Information")

# Create form inputs
with st.sidebar.form("loan_form"):
    # Personal Information
    gender = st.selectbox("Gender", options=["Male", "Female"])
    married = st.selectbox("Marital Status", options=["Yes", "No"])
    dependents = st.selectbox("Number of Dependents", options=["0", "1", "2", "3+"])
    education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", options=["Yes", "No"])
    
    # Financial Information
    applicant_income = st.number_input("Applicant's Monthly Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Co-applicant's Monthly Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1, value=100)
    loan_term = st.slider("Loan Term (in months)", min_value=12, max_value=480, value=360, step=12)
    
    # Credit Information
    credit_history = st.selectbox("Has Credit History", options=["Yes", "No"])
    credit_history_binary = 1 if credit_history == "Yes" else 0
    
    # Property Information
    property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])
    
    # Submit button
    submit_button = st.form_submit_button("Predict Loan Approval")

# Main content area
if "submit_button" not in locals() or submit_button:
    # Prepare input data
    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history_binary,
        'Property_Area': property_area
    }
    
    # Load model and columns
    model, model_columns = load_model_and_columns()
    
    if model is None:
        st.warning("Model not found. Using a simplified rule-based prediction for demonstration purposes only.")
        
        # Simple rule-based prediction for demonstration
        approval_chance = 0
        
        # Credit history is very important
        if credit_history_binary == 1:
            approval_chance += 50
        
        # Income to loan ratio
        total_income = applicant_income + coapplicant_income
        if total_income / loan_amount > 5:
            approval_chance += 30
        elif total_income / loan_amount > 2:
            approval_chance += 20
        else:
            approval_chance += 10
        
        # Education factor
        if education == "Graduate":
            approval_chance += 10
        
        # Property area factor
        if property_area == "Urban" or property_area == "Semiurban":
            approval_chance += 10
        
        # Clamp the value
        approval_chance = min(approval_chance, 95)
        
        # Display prediction
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Loan Approval Prediction")
            if approval_chance > 70:
                st.success("Loan Approval: LIKELY TO BE APPROVED")
            else:
                st.error("Loan Approval: LIKELY TO BE REJECTED")
        
        with col2:
            st.subheader("Approval Chance")
            st.info(f"Estimated Approval Chance: {approval_chance}%")
            st.caption("Note: This is a simulated prediction for demonstration only.")
        
        # Display factors affecting the decision
        st.subheader("Key Factors Affecting Decision")
        
        factors = []
        if credit_history_binary == 1:
            factors.append("✅ Good credit history")
        else:
            factors.append("❌ No credit history")
            
        if total_income / loan_amount > 5:
            factors.append("✅ Excellent income to loan ratio")
        elif total_income / loan_amount > 2:
            factors.append("✅ Good income to loan ratio")
        else:
            factors.append("❌ Low income to loan ratio")
            
        if education == "Graduate":
            factors.append("✅ Graduate education")
        else:
            factors.append("ℹ️ Non-graduate education")
            
        st.write(", ".join(factors))
        
    else:
        # Preprocess input data
        processed_input = preprocess_input(input_data)
        
        # Make sure test data has the same columns as training data
        if model_columns is not None:
            # Check for missing columns
            missing_cols = set(model_columns) - set(processed_input.columns)
            # Add missing columns with default value of 0
            for col in missing_cols:
                processed_input[col] = 0
            # Ensure the order of columns is correct
            processed_input = processed_input[model_columns]
        
        # Make prediction
        try:
            prediction = model.predict(processed_input)[0]
            prediction_proba = model.predict_proba(processed_input)[0]
            
            # Display prediction
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Loan Approval Prediction")
                if prediction == 1:
                    st.success("Loan Approval: APPROVED")
                else:
                    st.error("Loan Approval: REJECTED")
            
            with col2:
                st.subheader("Confidence")
                approval_probability = prediction_proba[1] * 100
                st.info(f"Approval Confidence: {approval_probability:.2f}%")
                
            # Feature importance (if using XGBoost)
            if hasattr(model, 'feature_importances_'):
                st.subheader("Key Factors Influencing Decision")
                
                # Credit history is generally most important
                factors = []
                if credit_history_binary == 1:
                    factors.append("✅ Good credit history")
                else:
                    factors.append("❌ No credit history")
                    
                # Income to loan ratio
                total_income = applicant_income + coapplicant_income
                if total_income / loan_amount > 5:
                    factors.append("✅ Excellent income to loan ratio")
                elif total_income / loan_amount > 2:
                    factors.append("✅ Good income to loan ratio")
                else:
                    factors.append("❌ Low income to loan ratio")
                    
                st.write(", ".join(factors))
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("There may be a mismatch between the model's expected features and the input data. Try running 'save_model.py' again.")
                
# Add information about the model
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("""
This app uses a machine learning model to predict whether a loan application will be approved 
based on various factors like income, credit history, and more. 

The prediction is based on historical data and should be used as a guide only.
""")

# Add instructions
st.sidebar.markdown("---")
st.sidebar.subheader("Instructions")
st.sidebar.info("""
1. Fill in all the details in the form.
2. Click on 'Predict Loan Approval'.
3. The prediction result will be displayed.

For best results, ensure all information is accurate.
""")

# Add data description
with st.expander("Data Description"):
    st.markdown("""
    | Feature | Description |
    | --- | --- |
    | Gender | Male / Female |
    | Married | Marital status (Yes/No) |
    | Dependents | Number of family members |
    | Education | Graduate / Not Graduate |
    | Self_Employed | Self-employed (Yes/No) |
    | ApplicantIncome | Monthly income |
    | CoapplicantIncome | Co-applicant's monthly income |
    | LoanAmount | Loan amount in thousands |
    | Loan_Amount_Term | Term of loan in months |
    | Credit_History | Past credit history (1: Good, 0: Bad) |
    | Property_Area | Urban / Semiurban / Rural |
    """)

# Add setup instructions
with st.expander("Setup Instructions"):
    st.markdown("""
    ### How to set up this application
    
    1. Install required packages:
       ```
       pip install -r requirements.txt
       ```
       
    2. Train and save the model:
       ```
       python save_model.py
       ```
       
    3. Run the Streamlit app:
       ```
       streamlit run loan_prediction_app.py
       ```
    """)

# Footer
st.markdown("---")
st.caption("© 2023 Loan Prediction System - Created for demonstration purposes only") 