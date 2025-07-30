from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
import pickle

HR_df_encoded_transformed_scaled = pd.read_pickle(
    "../../HR_Analytics/data/processed/HR_df_encoded_transformed_scaled.pkl"
)
# --------------------   ------------------------------------------------
# splitting the dataset into training and testing sets
# --------------------------------------------------------------------
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
    "JobRole_Laboratory Technician",
]
X = HR_df_encoded_transformed_scaled[selected_features]  # explanatory variables
y = HR_df_encoded_transformed_scaled["Attrition_Yes"]  # target/response variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.to_pickle("../../HR_Analytics/data/interim/X_train.pkl")
X_test.to_pickle("../../HR_Analytics/data/interim/X_test.pkl")
y_train.to_pickle("../../HR_Analytics/data/interim/y_train.pkl")
y_test.to_pickle("../../HR_Analytics/data/interim/y_test.pkl")

# --------------------------------------------------------------------
# Ensemble method Logistic regression training script
# --------------------------------------------------------------------


def fit_logistic_model(features_train, target_train, iterations):
    logreg = LogisticRegression(max_iter=iterations, random_state=42)
    logreg.fit(features_train, target_train)
    return logreg


logreg = fit_logistic_model(X_train, y_train, 1000)

# Save the trained logistic regression model using pickle
with open("../../HR_Analytics/models/logistic_regression_model.pkl", "wb") as f:
    pickle.dump(logreg, f)
    



# --------------------------------------------------------------------
# Random Forest training script
# --------------------------------------------------------------------


def random_forest_model(features_train, target_train, n_estimators=500, max_depth=10):
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42, class_weight='balanced'
    )
    rf.fit(features_train, target_train)
    return rf

rf_model = random_forest_model(X_train, y_train)

# Save the trained random forest model using pickle
with open("../../HR_Analytics/models/random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# --------------------------------------------------------------------
# XGBoost training script
# --------------------------------------------------------------------
