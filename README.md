# Loan Eligibility Prediction

This project is a machine learning application that predicts whether an individual is eligible for a loan based on various features such as income, education, and credit history. The project uses Python and popular libraries like NumPy, Pandas, and Scikit-learn to preprocess the data, build the model, and evaluate its performance.

## Table of Contents
1. [Overview](#overview)
2. [Technologies Used](#technologies-used)
3. [Dataset](#dataset)
4. [Preprocessing Steps](#preprocessing-steps)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Prediction](#prediction)

---

## Overview
This project focuses on predicting the loan eligibility of an individual. The dataset includes features such as marital status, gender, income, loan amount, and more. 

The model is built using a Support Vector Machine (SVM) with a linear kernel, and its accuracy is measured on both training and test data.

---

## Technologies Used
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Seaborn (optional for visualization)

---

## Dataset
The dataset used for this project contains information about individuals and their loan application status. Some key features:
- **Loan_ID**: Unique identifier for each loan application.
- **Gender**: Male (1) or Female (0).
- **Married**: Yes (1) or No (0).
- **Education**: Graduate (1) or Not Graduate (0).
- **ApplicantIncome**: Monthly income of the applicant.
- **CoapplicantIncome**: Monthly income of the co-applicant.
- **LoanAmount**: Loan amount requested.
- **Loan_Amount_Term**: Term of the loan in months.
- **Credit_History**: Credit history record (1 for clear, 0 for unclear).
- **Property_Area**: Location of the property (Rural: 0, Semiurban: 1, Urban: 2).
- **Loan_Status**: Loan approved (1) or not approved (0).

---

## Preprocessing Steps
1. **Handling Missing Values**:
   - Missing values are removed using `dropna()`.
2. **Encoding Categorical Variables**:
   - Replaced categorical variables with numerical equivalents, e.g., `Yes`/`No` -> `1`/`0`.
3. **Replacing LoanAmount values**:
   - Replaced "3+" in `Dependents` with `4`.
4. **Feature Selection**:
   - Removed `Loan_ID` and `Loan_Status` columns from the feature set.

---

## Model Training and Evaluation
1. **Splitting Data**:
   - The dataset is split into training and test sets (90% training, 10% testing).
2. **Training the Model**:
   - An SVM with a linear kernel is trained on the data.
3. **Evaluating Performance**:
   - **Training Data Accuracy**: `80%` is used to measure performance on the training data.
   - **Test Data Accuracy**: `83%`

---

## Prediction
The model predicts loan eligibility based on new input data.

Eligible for loan: If the prediction is 1.

Not eligible for loan: If the prediction is 0.

### Example Input:
```python
input_data = (1,1,1,1,0,4583,1508,128,360,1,0)
```
## Results
Training Data Accuracy:

Achieved accuracy on training data: `80%`

Test Data Accuracy:

Achieved accuracy on test data: `83%`

## Future Enhancements

Add data visualizations for better insight.

Optimize the model by testing different algorithms.

Include more advanced preprocessing techniques for better accuracy.


