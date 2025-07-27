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


