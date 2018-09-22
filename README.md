# Spot-Check-RF-Vs-AutoARIMA

_

### Prerequisites

_

### Installing

_

### Goal

Compare the RMSE of AutoARIMA with RF for a 10 day forecast.

### Data Sources

- https://datamarket.com/data/set/235k/daily-total-female-births-in-california-1959#!ds=235k&display=line
- daily-total-female-births-in-cal.csv

### Data Descriptions

- Date: The day without a timestamp
- Daily total female births in California, 1959

### Steps

- Start
  - X Assume it is stationary to start (only go back to this if stuck)
- Spot Check Algorithms
  - Train/test split last 10 days
  - Try Auto ARIMA, get the RMSE over 10 days
  - Try RF
    - Train 10 models, each X days ahead (copy code)
    - Use parameters from another project
    - The test set should be the last available record and 1-10 days ahead (one day prediction for 10 models)
