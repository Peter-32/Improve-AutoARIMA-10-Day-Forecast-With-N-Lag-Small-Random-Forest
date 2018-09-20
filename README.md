# Spot-Check-RF-Vs-AutoARIMA

_

### Prerequisites

_

### Installing

_

### Goal

Run a 10 small random forest times on a univariate time series to predict the next 10 days.  Each model predicts one of the days ahead.  Compare the RMSE to AutoARIMA.  View the plots of each

### Data Sources

- https://www.kaggle.com/felixzhao/productdemandforecasting/home
- historical_product_demand.csv

### Data Descriptions

- Descriptions from: https://www.kaggle.com/felixzhao/productdemandforecasting/home
- Product_CodeThe product name encoded
- WarehouseWarehouse name encoded
- Product_CategoryProduct Category for each Product_Code encoded
- DateThe date customer needs the product
- Order_Demandsingle order qty


### Steps

- Use a text editor
- Load the data into Pandas
- Choose 10 models with the most records
- Check for missing values / infinity
- View the ten plots
- Consider capping at 95% percentiles and/or logging data
- Optionally cap the data at 95% percentiles
- View the ten plots
- Optionally log the data
- View the ten plots
- Consider if it looks stationary; regardless don't try to fix it
- Use last 10 days as the test set
- Check the average and persistence RMSE
- Check Auto ARIMA RMSE
- Check lag 8 Random Forest RMSE
- Record all results in a pandas then CSV file
- Print out the prediction for all 10 plots 1.png, ... 10.png.  One line for actual, AUTOArima, and RF
- Title the plots as the RMSE for each line, and a line for the best RMSE.  Add a legend.  Include a baseline RMSE in this title.
- End of project
