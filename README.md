# Spot-Check-RF-Vs-AutoARIMA

_

### Prerequisites

_

### Installing

_

### Question

Is the RF with 8 lags and a small number of components a better model than AutoARIMA for one dataset?

### Hypothesis

RF will have the better score.

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
  - X Train/test split last 10 days
  - X Try Auto ARIMA, get the RMSE over 10 days
  - Try RF
    - X lags
    - X Train 10 models, each X days ahead (copy code)
    - X Use parameters from another project
    - X The test set should be the last available record and 1-10 days ahead (one day prediction for 10 models)

### Conclusion

The hypothesis that RF will have a better score was correct.  It outperformed ARIMA drastically on the test set.  Random Forest has an RMSE of 4.04 and ARIMA has an RMSE of 7.64.
