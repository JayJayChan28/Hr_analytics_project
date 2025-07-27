import pandas as pd 
from sklearn.discriminant_analysis import StandardScaler
from sklearn.utils import resample
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



#--------------------------------------------------------------------
# Preprocessing the HR dataset
#--------------------------------------------------------------------

### define functions for testing different transformations
def log_trans(df, columns):
    for features in columns:
        df[f'{features}_log'] = np.log1p(df[features])
    return df

def sqrt_trans(df, columns):
    for features in columns:
        df[f'{features}_sqrt'] = np.sqrt(df[features])
    return df

def cbrt_trans(df, columns):
    for features in columns:
        df[f'{features}_cbrt'] = np.cbrt(df[features])
    return df

#--------------------------------------------------------------------
# apply non-linear transformation to the numeric features
#--------------------------------------------------------------------
### monthly income, years at company, total working hours and distance from home are very right scewed we want to normalize this so that it looks more like a bell shaped curve
### attempt log transformation
HR_df_log = log_trans(HR_df, ['MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears', 'DistanceFromHome'])
HR_df_log.to_pickle("../../HR_Analytics/data/interim/HR_df_log.pkl")

### Distance from home_log and Years at company_log didnt produce a bell shaped curve
### attempt sqrt transformation
HR_df_log_sqrt = HR_df_log.copy().drop(columns=['DistanceFromHome_log', 'YearsAtCompany_log'], errors='ignore')
HR_df_log_sqrt = sqrt_trans(HR_df_log_sqrt, ['DistanceFromHome', 'YearsAtCompany'])
HR_df_log_sqrt.to_pickle("../../HR_Analytics/data/interim/HR_df_log_sqrt.pkl")

### SQRT transformation not working for distance from home, CBRT transformation works better
HR_df_log_sqrt_crbt = HR_df_log_sqrt.copy().drop(columns=['DistanceFromHome_sqrt'], errors='ignore')
HR_df_log_sqrt_crbt = cbrt_trans(HR_df_log_sqrt_crbt, ['DistanceFromHome'])
HR_df_log_sqrt_crbt.to_pickle("../../HR_Analytics/data/interim/HR_df_log_sqrt_crbt.pkl")

### Drop the original features that were transformed
HR_df_transformed = HR_df_log_sqrt_crbt.copy().drop(columns=['MonthlyIncome', 'YearsAtCompany', "TotalWorkingYears", 'DistanceFromHome'])
HR_df_transformed.to_pickle("../../HR_Analytics/data/interim/HR_df_transformed.pkl")
#--------------------------------------------------------------------
# Scale/Normalize the numeric features
#--------------------------------------------------------------------

# The following features are selected for scaling because they are numeric and may have different ranges;
# scaling helps to standardize these features, improving model convergence and performance.
features_to_scale = [
    "Age",
    "EmployeeCount",
    "DailyRate",
    "EmployeeNumber",
    "HourlyRate",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "StandardHours",
    "StockOptionLevel",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "MonthlyIncome_log",
    "TotalWorkingYears_log",
    "DistanceFromHome_cbrt",
    "YearsAtCompany_sqrt"
]


scaler = StandardScaler()
HR_df_transformed_scaled = HR_df_transformed.copy()
HR_df_transformed_scaled[features_to_scale] = scaler.fit_transform(HR_df_transformed_scaled[features_to_scale])

HR_df_transformed_scaled.to_pickle("../../HR_Analytics/data/interim/HR_df_transformed_scaled.pkl")




#--------------------------------------------------------------------
# One hot encoding categorical features
#--------------------------------------------------------------------
One_hot_encoded_columns = [
    "Attrition",
    "BusinessTravel",
    "Gender",
    "EducationField",
    "JobRole",
    "Department",
    "MaritalStatus",
    "OverTime",
    "Over18",
]
HR_df_encoded_transformed_scaled = pd.get_dummies(HR_df_transformed_scaled, columns=One_hot_encoded_columns, drop_first=True)   
HR_df_encoded_transformed_scaled.to_pickle("../../HR_Analytics/data/processed/HR_df_encoded_transformed_scaled.pkl")

#--------------------------------------------------------------------
# Upsampling miniority class (oversampling)
#--------------------------------------------------------------------
HR_df.query("Attrition == 'Yes'").shape[0] # number of employees who left the company
HR_df.query("Attrition == 'No'").shape[0] # number of employees who stayed  

### Minority class inbalanced, we will upsample the minority class (Attrition = 1) to balance the dataset

HR_df_upsampled 
