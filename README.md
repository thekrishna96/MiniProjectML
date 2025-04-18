# Loan Approval Prediction Using Machine Learning

This project predicts whether a loan application should be approved based on the applicant's profile, including income, credit history, employment status, and other factors.

## Dataset

The dataset used in this project is the [Loan Prediction Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) from Kaggle, which contains information about loan applicants and whether their applications were approved or rejected.

### Features

| Column Name       | Description                             |
| ----------------- | --------------------------------------- |
| Loan_ID           | Unique Loan ID                          |
| Gender            | Male / Female                           |
| Married           | Applicant married (Yes/No)              |
| Dependents        | Number of dependents                    |
| Education         | Graduate / Not Graduate                 |
| Self_Employed     | Self-employed? (Yes/No)                 |
| ApplicantIncome   | Monthly income of applicant             |
| CoapplicantIncome | Monthly income of co-applicant          |
| LoanAmount        | Loan amount (in thousands)              |
| Loan_Amount_Term  | Term of loan in months                  |
| Credit_History    | 1 (Yes), 0 (No)                         |
| Property_Area     | Urban / Semiurban / Rural               |
| Loan_Status       | Target — Y (Approved), N (Not Approved) |

## Project Structure

- `train.csv` - Training dataset
- `test.csv` - Test dataset
- `loan_prediction.py` - Basic ML models (Logistic Regression & Random Forest)
- `loan_prediction_xgboost.py` - Advanced ML model with XGBoost and hyperparameter tuning
- `loan_prediction_app.py` - Streamlit web application for loan prediction
- `requirements.txt` - Required packages for the project

## Implementation

This project implements three machine learning algorithms for loan prediction:

1. **Logistic Regression** - A basic classification algorithm suitable for binary outcomes
2. **Random Forest** - An ensemble method that works well with categorical and numerical features
3. **XGBoost** - A gradient boosting algorithm known for its high performance

### Data Preprocessing

The data preprocessing steps include:

- Handling missing values
- Feature engineering (creating combined features like total income, EMI calculations)
- Encoding categorical variables
- Log transformation for skewed numerical features

### Model Evaluation

Models are evaluated using:

- Accuracy, Precision, Recall, F1-Score
- ROC Curve and AUC
- Confusion Matrix
- Cross-validation

## Usage

### Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

### Running the Models

1. To run the basic models (Logistic Regression & Random Forest):

```bash
python loan_prediction.py
```

2. To run the XGBoost model with hyperparameter tuning:

```bash
python loan_prediction_xgboost.py
```

3. To launch the Streamlit web application:

```bash
streamlit run loan_prediction_app.py
```

## Web Application

The Streamlit web application allows users to:

1. Input applicant details
2. Get predictions on loan approval
3. View the confidence of predictions
4. Understand key factors affecting the decision

## Results and Insights

After evaluating the models, the following insights were obtained:

- Credit history is the most important factor for loan approval
- Income to loan amount ratio significantly affects the approval decision
- XGBoost generally provides the best prediction accuracy

## Future Improvements

Potential improvements for this project include:

- Using more sophisticated feature engineering techniques
- Addressing class imbalance with techniques like SMOTE
- Implementing a deployment pipeline for production use

## License

This project is licensed under the MIT License.

## Acknowledgements

- The dataset is from [Kaggle's Loan Prediction Problem](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- The project is created for educational purposes
