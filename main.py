import pandas as pd
# import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# import missingno as msno
from pandasql import sqldf
from scipy.stats import mstats
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
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

### FUNCTIONS - ETL
def do_etl():
    df = pd.read_csv("source_data/historical_product_demand.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df = sqldf("""SELECT product_code
                       || "-"
                       || warehouse
                       || "-"
                       || product_category AS model_key,
                       date,
                       Sum(IFNULL(order_demand, 0))   AS Order_Demand
                FROM   df
                GROUP  BY 1,
                          2 """, locals())
    df = sqldf("""SELECT a.*
                FROM   df a
                       INNER JOIN (SELECT model_key,
                                          Count(*) n
                                   FROM   df
                                   GROUP  BY 1
                                   ORDER  BY 2 DESC,
                                             1
                                   LIMIT  10) b
                               ON b.model_key = a.model_key
                ORDER  BY model_key,
                          Date """, locals())
    df.to_csv("tables/1_ten_models.csv", index=False)

### FUNCTIONS - Exploration
def do_exploration():
    df = pd.read_csv("tables/1_ten_models.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(["Date"])
    ten_plots(df=df, filename="1_original_data")
    # Good min/max dates: 1/5/12 and 12/28/16
    # Will cap at 95% upper bound and 5% lower bound
    # Will not log the data
    # Will impute using 7 day moving average
    # There are lots of missing values
    print("There are infinity values: {}".format(df.fillna(0).replace([np.inf, -np.inf], np.nan).isnull().values.any()))
    # There are no infinity values

def ten_plots(df, filename):
    df = pd.pivot_table(data=df, values='Order_Demand', index="Date", columns="model_key")
    df.to_csv("tables/_row_date__col_product.csv")
    colors=['#A8E6CE', '#A8E6CE',
            '#DCEDC2', '#DCEDC2',
            '#FFD3B5', '#FFD3B5',
            '#FFAAA6', '#FFAAA6',
            '#FF8C94',  '#FF8C94']
    fig, ax = plt.subplots(nrows=5, ncols=2)
    for i, ax in enumerate(ax.flatten()):
        df[df.columns[i]].plot(color=colors[i], ax=ax, sharex=ax)
        ax.set_title(df.columns[i], fontsize=8)
    plt.savefig('images/{}.png'.format(filename))

### FUNCTIONS - Prepare Data
def do_prepare_data():
    main_df = pd.read_csv("tables/1_ten_models.csv")
    model_keys=main_df['model_key'].unique().tolist()
    prepared_df = pd.DataFrame()
    for model_key in model_keys:
        df = load_data(main_df, model_key=model_key)
        df = fill_missing_days__and_set_datetime_index(df, start_date="2012-01-05", end_date="2016-12-28")
        df.to_csv("tables/_fill_days.csv")
        df.loc[:,'Order_Demand'] = mstats.winsorize(df['Order_Demand'].values, limits=[0.05, 0.05])
        df = moving_average_imputation(df)
        plot(df, model_key, '2_imputation_example')
        df['model_key'] = df['Order_Demand'].apply(lambda x: model_key)
        df['Date'] = df.index
        prepared_df = pd.concat([prepared_df, df])
    prepared_df = sqldf("select model_key, Date, Order_Demand from prepared_df ORDER BY 1,2", locals())
    prepared_df.to_csv("tables/2_data_prepared.csv")
    prepared_df.set_index(['Date'], inplace=True)
    ten_plots(df=prepared_df, filename="3_prepared_data")

def load_data(main_df, model_key):
    return sqldf("SELECT Date, Order_Demand from main_df where model_key = '{}'".format(model_key), locals())

def fill_missing_days__and_set_datetime_index(df, start_date, end_date):
  idx = pd.date_range(start_date, end_date)
  df['Date'] = pd.to_datetime(df['Date'])
  df.set_index('Date', inplace=True)
  df = df.reindex(idx, fill_value=0)
  return df

def moving_average_imputation(df):
    # Consider +/- infinity here if working on a different time series
    df = df.replace(0, np.nan)
    df['Order_Demand'] = df['Order_Demand'].fillna(df['Order_Demand'].rolling(window=7, min_periods=1, center=False).mean())
    df['Order_Demand'] = df['Order_Demand'].fillna(method='ffill')
    return df

def plot(df, model_key, filename):
    fig, ax = plt.subplots()
    ax.set_title(model_key, fontsize=12)
    plt.plot(df['Order_Demand'])
    plt.savefig('images/{}.png'.format(filename))

##### MAIN

### ETL
# do_etl()

### Exploration
# do_exploration()

### Prepare data
# do_prepare_data()








#
# print(df.head())
# print(df.dtypes)
# print(df.info())
# print(df.index)
