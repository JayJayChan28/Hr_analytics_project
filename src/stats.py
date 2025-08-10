import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
pd.set_option('display.max_columns', None)
HR_df = pd.read_pickle("../../HR_Analytics/data/interim/HR_df.pkl")
HR_df_encoded_transformed_scaled = pd.read_pickle("../../HR_Analytics/data/processed/HR_df_encoded_transformed_scaled.pkl")

#--------------------------------------------------------------------
# Descriptive Stastistics
#--------------------------------------------------------------------

HR_df.info()# information about the dataset, all dtypes are correct
HR_df.describe() # statistical summary of the dataset
HR_df.isnull().sum() # shows the number of missing values in each column

HR_df[HR_df.duplicated()].shape[0] # shows the number of duplicate rows in the dataset
#--------------------------------------------------------------------
# Correlation Analysis
#--------------------------------------------------------------------
corr_matrix = HR_df_encoded_transformed_scaled.corr()
corr_matrix['Attrition_Yes'].sort_values(ascending=False) # correlation of each feature with Attrition (Yes)
### Linear relationship looks weak try logistic regression analysis

def fit_logistic_model(df, target):
    # Select all numeric columns
    X = df.drop(target, axis=1)
    X = sm.add_constant(X)  # adds intercept
    X = X.astype(float)
    for col in X.columns:
        if X[col].nunique() < 2:
            X.drop(col, axis=1, inplace=True) # ensure all features have more than one unique value to avoid singular matrix errors
    y = df[target].astype(float)  # ensure target is float

    model = sm.Logit(y, X).fit()
    return model

model = fit_logistic_model(HR_df_encoded_transformed_scaled, 'Attrition_Yes')
print(model.summary())
#now we select the features with p-values less than 0.05 to identify significant predictors
for feature, pvalue in model.pvalues.sort_values(ascending=True).items():
    selected_features = []
    if pvalue < 0.05:  # significance level
        print(f"Feature: {feature}, P-Value: {pvalue}")
        selected_features.append(feature)

selected_features = [
    "OverTime_Yes",
    "EnvironmentSatisfaction",
    "NumCompaniesWorked",
    "JobSatisfaction",
    "BusinessTravel_Travel_Frequently",
    "JobInvolvement",
    "YearsSinceLastPromotion",
    "DistanceFromHome_cbrt",
    "TotalWorkingYears_log",
    "RelationshipSatisfaction",
    "MaritalStatus_Single",
    "WorkLifeBalance",
    "BusinessTravel_Travel_Rarely",
    "TrainingTimesLastYear",
    "Gender_Male",
    "MonthlyIncome_log",
    "JobRole_Laboratory Technician"
]