#--------------------------------------------------------------------
# importing libraries
#--------------------------------------------------------------------

import pandas as pd 
from sklearn.discriminant_analysis import StandardScaler
from sklearn.utils import resample
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

HR_df = pd.read_csv("../data/raw/HR-Employee-Attrition.csv")
HR_df.to_pickle("../../HR_Analytics/data/interim/HR_df.pkl")
#--------------------------------------------------------------------
# Preprocessing the HR dataset
#--------------------------------------------------------------------

HR_df.info()# information about the dataset
HR_df.describe() # statistical summary of the dataset
HR_df.isnull().sum() # shows the number of missing values in each column

HR_df[HR_df.duplicated()].shape[0] # shows the number of duplicate rows in the dataset
#--------------------------------------------------------------------
# apply non-lienar transformation to the numeric features
#--------------------------------------------------------------------

### monthly income, years at company, total working hours and distance from home are very right scewed we want to normailizat this so that it looks more like a bell shaped curve
HR_df_log = HR_df.copy()
HR_df_log['MonthlyIncome_log'] = np.log1p(HR_df['MonthlyIncome'])
HR_df_log['YearsAtCompany_log'] = np.log1p(HR_df['YearsAtCompany'])
HR_df_log['TotalWorkingHours_log'] = np.log1p(HR_df['TotalWorkingYears'])
HR_df_log['DistanceFromHome_log'] = np.log1p(HR_df['DistanceFromHome'])
HR_df_log.to_pickle("../../HR_Analytics/data/interim/HR_df_log.pkl")

### Distance from home_log and Years at company_log didnt produce a bell shaped curve, try sqrt transformation on those features
HR_df_log_sqrt = HR_df_log.copy().drop(columns=['DistanceFromHome_log', 'YearsAtCompany_log'])
HR_df_log_sqrt['DistanceFromHome_cbrt'] = np.cbrt(HR_df_log['DistanceFromHome_log'])
HR_df_log_sqrt['YearsAtCompany_sqrt'] = np.sqrt(HR_df_log['YearsAtCompany_log'])
HR_df_log_sqrt.to_pickle("../../HR_Analytics/data/interim/HR_df_log_sqrt.pkl")

### SQRT transformation not working for distance from home, CBRT transformation works better

### Drop the original features that were transformed
HR_df_transformed = HR_df_log_sqrt.copy().drop(columns=['MonthlyIncome', 'YearsAtCompany', "TotalWorkingYears", 'DistanceFromHome'])
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
    "TotalWorkingHours_log",
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


