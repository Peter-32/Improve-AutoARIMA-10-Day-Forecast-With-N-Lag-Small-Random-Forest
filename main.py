import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandasql import sqldf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pyramid.arima import auto_arima
from sklearn.metrics import mean_squared_error
from sklearn import ensemble


def q(q): return sqldf(q, globals())

# Prepare Data


def get_train_test_split(df, days_in_test_set):
    days_in_test_set = days_in_test_set
    split_point = len(df) - days_in_test_set
    train, test = df[0:split_point], df[split_point:]
    return train, test


df = pd.read_csv(
    "/Users/peterjmyers/Work/Spot-Check-RF-Vs-AutoARIMA/source_data/daily-total-female-births-in-cal.csv")
df = df.iloc[0:365]
df.columns = ['Date', 'x']
df['Date'] = pd.to_datetime(df['Date'])
df.index = df['Date']
df.drop(['Date'], axis=1, inplace=True)
days_in_test_set = 10
train, test = get_train_test_split(df, days_in_test_set)
train, test

# Improve Algorithm


def rmse(y_pred, y_true):
    return np.sqrt(mean_squared_error(y_pred, y_true))


def auto_arima_prediction(train, days_in_test_set):
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
    return arima_y_pred


def random_forest_prediction(df, days_in_test_set):
    for i in range(1, 9):
        df['lag_{}'.format(i)] = df['x'].shift(i)
    df = df.dropna()
    for forecast_days_ahead in range(1, 11):
        df['y_{}'.format(forecast_days_ahead)
           ] = df['x'].shift(-1 * forecast_days_ahead)
    train, test = get_train_test_split(df, days_in_test_set)

    train_X = train[['x', 'lag_1', 'lag_2', 'lag_3',
                     'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8']]
    test_X = train.iloc[-1][['x', 'lag_1', 'lag_2', 'lag_3',
                             'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8']]
    test_X
    rf_pred_y = []
    for days_ahead in range(1, 11):
        model = ensemble.RandomForestRegressor(
            n_estimators=50, max_features="log2", min_samples_leaf=5, criterion="mse", bootstrap=True, random_state=2)
        train_y = train['y_{}'.format(days_ahead)]
        model = model.fit(train_X, train_y)
        rf_pred_y.append(model.predict(test_X))
    rf_pred_y = np.array(rf_pred_y)
    return rf_pred_y


arima_y_pred = auto_arima_prediction(train, days_in_test_set)
rf_pred_y = random_forest_prediction(df, days_in_test_set)

print("The AutoARIMA rmse is {}".format(rmse(test['x'], arima_y_pred)))
print("The Random Forest rmse is {}".format(rmse(test['x'], rf_pred_y)))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(test['x'].values)
ax.plot(arima_y_pred)
ax.plot(rf_pred_y)

ax.legend(['y_test', 'arima_y_pred', 'rf_pred_y'],
          loc='lower left')
