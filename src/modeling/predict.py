import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from src.modeling.train import X_test
with open("../../HR_Analytics/models/logistic_regression_model.pkl", "rb") as f:
    logreg = pickle.load(f)
with open("../../HR_Analytics/models/random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
X_test = pd.read_pickle("../../HR_Analytics/data/interim/X_test.pkl")
y_test = pd.read_pickle("../../HR_Analytics/data/interim/y_test.pkl")

#--------------------------------------------------------------------
# Logistic regression Predict script
#--------------------------1------------------------------------------
y_pred_logistic = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_logistic)
print("Accuracy:", accuracy)
#Accuracy is sort of high with 87.4% a little suspicious lets check the Precision and recall
print(classification_report(y_test, y_pred_logistic))
### Report shows model performs well with majority class due to class imbalance

#--------------------------------------------------------------------
# Random Forest Predict script
#--------------------------------------------------------------------

y_pred_random_forest = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_random_forest)
print("Random Forest Accuracy:", accuracy_rf)
### accuracy around the same as logistic regression
print(classification_report(y_test, y_pred_random_forest))
### Performance on true class is much worse than logistic regression
### increased depth to capture more complex relationships and balances the classes with class_weight='balanced'
### when increasing the number of estimators, model became worse, suspect that it learned the majority class too well due to class imbalance
### best recall I could get is 0.08 for Attrition = 1, which is not good enough

