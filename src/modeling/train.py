from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import time
import pickle
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import classification_report
HR_df_encoded_transformed_scaled = pd.read_pickle(
    "../../HR_Analytics/data/processed/HR_df_encoded_transformed_scaled.pkl"
)
# -------------------- ------------------------------------------------
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
    X, y, test_size=0.2, stratify=y, random_state=42
)

# checking class balance
print(y_train.value_counts())
print(y_test.value_counts())

X_train.to_pickle("../../HR_Analytics/data/interim/X_train.pkl")
X_test.to_pickle("../../HR_Analytics/data/interim/X_test.pkl")
y_train.to_pickle("../../HR_Analytics/data/interim/y_train.pkl")
y_test.to_pickle("../../HR_Analytics/data/interim/y_test.pkl")

# --------------------------------------------------------------------
# Logistic regression training script
# --------------------------------------------------------------------
pipe_log = Pipeline(
    [
        (
            "oversample",
            RandomOverSampler(random_state=42),
        ),  # overampling to handle class imbalance
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)
pipe_log.get_params()
parameters_grid = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__penalty": ["l1", "l2"],
    "classifier__solver": ["liblinear", "saga"],
    "classifier__class_weight": [None, "balanced"],
}

scoring_log = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}
grid_search_log = GridSearchCV(pipe_log, parameters_grid, cv=5, scoring=scoring_log, refit="recall")

start_time = time.time()
grid_search_log.fit(X_train, y_train)
end_time = time.time()

print(f"Logistic Regression training time: {end_time - start_time} seconds")
print("Best precision:", grid_search_log.cv_results_['mean_test_precision'][grid_search_log.best_index_])
print("Best recall:", grid_search_log.cv_results_['mean_test_recall'][grid_search_log.best_index_])
print("Best F1-score:", grid_search_log.cv_results_['mean_test_f1'][grid_search_log.best_index_])
# f1 score: 0.52
# recall 0.75
# Precision 0.4


# --------------------------------------------------------------------
# Random Forest training script
# --------------------------------------------------------------------

pipe_forest = Pipeline(
    [
        (
            "oversample",
            RandomOverSampler(random_state=42),
        ),  # overampling to handle class imbalance
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)
pipe_forest.get_params()

parameters_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__class_weight": [None, "balanced"],
    "classifier__max_features": ["sqrt", "log2"],
}
scoring_forest = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}
grid_search_forest = GridSearchCV(
    pipe_forest, parameters_grid, cv=5, scoring=scoring_forest, refit="recall"
)
start_time = time.time()
grid_search_forest.fit(X_train, y_train)
end_time = time.time()
print(f"Random Forest training time: {end_time - start_time} seconds")
print("Best precision:", grid_search_forest.cv_results_['mean_test_precision'][grid_search_forest.best_index_])
print("Best recall:", grid_search_forest.cv_results_['mean_test_recall'][grid_search_forest.best_index_])
print("Best F1-score:", grid_search_forest.cv_results_['mean_test_f1'][grid_search_forest.best_index_])