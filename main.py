import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import StandardScaler
from pandasql import sqldf
from sklearn.metrics import mean_squared_error
from sklearn import linear_model, neighbors, tree, svm, ensemble
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import boxcox
from scipy.special import inv_boxcox
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', 60)
%matplotlib inline
q = lambda q: sqldf(q, globals())
mpl.rcParams['figure.figsize'] = (16.0, 16.0)

df = pd.read_csv("daily-total-female-births-in-cal.csv", header=None)
print(df)

# , names = ['CRIM','INDUS','CHAS','NOX','RM','AGE','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
