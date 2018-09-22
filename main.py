import pandas as pd
# import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# import missingno as msno
from pandasql import sqldf
from statsmodels.tsa.arima_model import ARIMA
# from scipy.stats import mstats
import statsmodels.api as sm
from pyramid.arima import auto_arima
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
# from sklearn import linear_model, neighbors, tree, svm, ensemble
# from xgboost import XGBRegressor
# from sklearn.pipeline import make_pipeline
# from tpot.builtins import StackingEstimator
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
# from sklearn.grid_search import GridSearchCV
# from sklearn.pipeline import Pipeline
# from scipy.stats import boxcox
# from scipy.special import inv_boxcox
q = lambda q: sqldf(q, globals())

days_in_test_set = 10

def get_train_test_split(df, days_in_test_set):
    days_in_test_set = days_in_test_set
    split_point = len(df) - days_in_test_set
    train, test = df[0:split_point], df[split_point:]
    return train, test

def rmse(y_pred, y_true):
    return np.sqrt(mean_squared_error(y_pred, y_true))


### Improve Algorithm
# df = pd.read_csv("source_data/daily-total-female-births-in-cal.csv")
df = pd.read_csv("/Users/peterjmyers/Work/Spot-Check-RF-Vs-AutoARIMA/source_data/daily-total-female-births-in-cal.csv")
df = df.iloc[0:365]
df.columns = ['Date', 'x']
df['Date'] = pd.to_datetime(df['Date'])
df.index = df['Date']
df.drop(['Date'], axis=1, inplace=True)
train, test = get_train_test_split(df, days_in_test_set)
train, test


stepwise_model = auto_arima(train, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
print(stepwise_model.aic())
stepwise_model.fit(train)
arima_y_pred = stepwise_model.predict(n_periods=days_in_test_set)
arima_y_pred

print("The AutoARIMA rmse is {}".format(rmse(test['x'], arima_y_pred)))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(test['x'].values)
ax.plot(arima_y_pred)
ax.legend(['y_test', 'y_pred'],
          loc='lower left')
