# HR Analytics: Predicting Employee Attrition

## Project Overview

This project applies machine learning techniques to predict employee attrition using a real-world HR dataset. The workflow includes data exploration, feature engineering, handling class imbalance, model training (Logistic Regression, Random Forest), hyperparameter tuning, and evaluation. The goal is to build a reliable, interpretable model to help organizations identify employees at risk of leaving and support proactive HR interventions.

## Problem Statement

Employee attrition poses significant challenges for organizations, impacting productivity, morale, and costs. Accurately predicting which employees are likely to leave enables proactive HR interventions and strategic planning. This project aims to develop robust machine learning models to identify employees at risk of attrition, focusing on maximizing recall to ensure that most true attrition cases are detected.

## Project Organization

This project follows a clear and reproducible structure inspired by the Cookiecutter Data Science template, with the following key directories and files:

```
HR_Analytics/
├── data/
│   ├── raw/         # Original, immutable data dump
│   ├── interim/     # Intermediate data that has been transformed
│   ├── processed/   # Final data sets for modeling
│   └── external/    # Data from third-party sources
├── notebooks/       # Jupyter notebooks for exploration and reporting
├── src/             # Source code for data processing, feature engineering, and modeling
│   ├── modeling/    # Model training and prediction scripts
│   └── services/    # (Optional) Service layer for deployment or APIs
├── models/          # Trained and serialized models
├── reports/         # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures/     # Generated graphics and figures
├── requirements.txt # Project dependencies
├── README.md        # Project overview and instructions
├── LICENCE          # Licensing information
└── .gitignore       # Files and directories to be ignored by git
```

- **data/**: Contains all data files, organized by their processing stage.
- **notebooks/**: Contains Jupyter notebooks for EDA, modeling, and reporting.
- **src/**: All source code, including scripts for data loading, feature engineering, modeling, and plotting.
- **models/**: Stores trained model files (e.g., .pkl files).
- **reports/**: Contains generated reports and visualizations.
- **requirements.txt**: Lists all Python dependencies needed to run the project.
- **README.md**: Project documentation and usage instructions.
- **.gitignore**: Specifies files/folders to exclude from version control.

This structure ensures clarity, reproducibility, and ease of collaboration for data science and machine learning projects.

## Workflow

1. **Data Loading & EDA**
   - Load and inspect the HR dataset.
   - Visualize distributions and relationships using histograms, boxplots, and pie charts.

2. **Data Preprocessing**
   - Handle missing values and outliers.
   - Apply log, square root, and cube root transformations to normalize skewed features.
   - Scale continuous features using `StandardScaler`.
   - One-hot encode categorical variables.

3. **Feature Selection**
   - Perform correlation analysis.
   - Use logistic regression and p-values to select significant predictors.

4. **Modeling**
   - Split data into training and test sets with stratification.
   - Address class imbalance using oversampling.
   - Train and tune Logistic Regression and Random Forest models with `GridSearchCV`.

5. **Evaluation**
   - Evaluate models using precision, recall, and F1-score.
   - Compare training and test performance to check for overfitting.
   - Select the best model based on recall and generalization.

6. **Results & Insights**
   - Logistic Regression selected for its high recall and balanced performance.
   - Random Forest can be used if higher precision is desired at the expense of recall.

## Key Files

- `notebooks/notebook.ipynb`: Main analysis and modeling notebook.
- `src/`: Source code for data processing, feature engineering, and modeling.
- `data/`: Raw and processed datasets.
- `reports/`: Visualizations and summary reports.

## How to Run

1. Clone the repository and install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Open `notebooks/notebook.ipynb` in Jupyter or VS Code.
3. Run the notebook cells sequentially to reproduce the analysis and results.

## Dependencies

- pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn, statsmodels

## Results

- **Best Model:** Logistic Regression (high recall, interpretable, no overfitting)
- **Business Value:** Enables HR teams to proactively identify and retain at-risk employees.