
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
#read pickle file
HR_df = pd.read_pickle("../../HR_Analytics/data/interim/HR_df.pkl")
HR_df_log = pd.read_pickle("../../HR_Analytics/data/interim/HR_df_log.pkl")
HR_df_log_sqrt = pd.read_pickle("../../HR_Analytics/data/interim/HR_df_log_sqrt.pkl")
# Histogram of numeric features
columns = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "EmployeeNumber",
    "HourlyRate",
    "MonthlyRate",
    "MonthlyIncome",
    "YearsAtCompany",
    "TotalWorkingYears"
]
HR_df[columns].hist(figsize=(12, 8), bins=50)
plt.show()

# Histogram of log tranformation features 
logged_features = [
    "MonthlyIncome_log",
    "YearsAtCompany_log",
    "DistanceFromHome_log",
    "TotalWorkingHours_log",
]
HR_df_log[logged_features].hist(figsize=(12, 8), bins=50)
plt.show()


#trying sqrt transformation for dsitance from home and years at company since those features are still very right scewed

logged_n_sqrt_features= [
    "MonthlyIncome_log",
     "YearsAtCompany_log",
    "DistanceFromHome_cbrt",
    "YearsAtCompany_sqrt"
]
HR_df_log_sqrt[logged_n_sqrt_features].hist(figsize=(12, 8), bins=50)
plt.show()


# boxplot for DailyRate by Attrition
sns.boxplot(x="Attrition", y="DailyRate", data=HR_df, color="skyblue")
plt.title("DailyRate by Attrition")
plt.show()
# looks like average DailyRate is lower for employees who left the company (Attrition = 1) compared to those who stayed (Attrition = 0)


# correlation heatmap




