# dataCampTimeSeriesAnalysis
class note for DataCamp

## Some Python Function

### Change an index to datetime

`df.index = pd.to_datetime(df.index)`

### Plotting data
#### Plot 2012 data using slicing
`df['2012'].plot()`

### Join data
`df1.join(df2)`
~~~~ 
# Import pandas
import pandas as pd

# Convert the stock index and bond index into sets
set_stock_dates = set(stocks.index)
set_bond_dates = set(bonds.index)

# Take the difference between the sets and print
print(set_stock_dates - set_bond_dates)

# Merge stocks and bonds DataFrames using join()
stocks_and_bonds = stocks.join(bonds,how= 'inner')
print (stocks_and_bonds.head())
~~~~


## Correlation of Two Time Series
The correlation coefficient is measure of how much two series vary together, correlation one means that the two series has a perfect relationship with on deviation. A low correlation means they vary together but there is a week association. And negative correlation means the vary in opposite direction, but still with a linear relationship.

#### excercise 1:
~~~~ 
# Compute percent change using pct_change()
returns = stocks_and_bonds.pct_change()

# Compute correlation using corr()
correlation = returns['SP500'].corr(returns['US10Y'])
print("Correlation of stocks and interest rates: ", correlation)

# Make scatter plot
plt.scatter(x = returns['SP500'],y=returns['US10Y'])
plt.show()
~~~~ 

#### excercise 2: levels vs percentage
~~~~ 
# Compute correlation of levels
correlation1 = levels.DJI.corr(levels.UFO)
print("Correlation of levels: ", correlation1)

# Compute correlation of percent changes
changes = levels.pct_change()
correlation2 = changes.DJI.corr(changes.UFO)
print("Correlation of changes: ", correlation2)
~~~~ 

## Simple Linear Regression

Regression also know as Ordinary Least Square(OLS)
~~~
import statsmodels.api as sm
df['SPX_Ret'] = df['SPX_Prices'].pct_change()
df['R2000_Ret'] = df['R2000_Prices'].pct_change()
# add a constant to the DataFrame for the regression intercept
df = sm.add_constant(df)
df = df.dropna()
results = sm.OLS(df['R2000_Ret'], df[['const', 'SPX_Ret']]).fit()
print(results.summary())
~~~
Regression Output:
* coef
Intercept = results.params[0]
Slope = results.params[1]
* R-Square
R-Square measures how well the linear regression line fits the data
[corr(x,y)]2 = R-square
sign(corr) = sign(regression slope)

## Looking at a Regression's R-squared
~~~
# Import the statsmodels module
import statsmodels.api as sm
# Compute correlation of x and y
correlation = y.corr(x)
print("The correlation between x and y is %4.2f" %(correlation))
# Convert the Series x to a DataFrame and name the column x
x = pd.DataFrame(x, columns=['x'])
# Add a constant to the DataFrame x
x = sm.add_constant(x)
# Fit the regression of y on x
result = sm.OLS(y,x).fit()
# Print out the results and look at the relationship between R-squared and the correlation 
~~~
The correlation is the square root of the R-squared, not the R-squared.

#Autocorrelation
correlation of a time series with a lagged copy of itself.
Negative Autocorrelation -> mean reverting
Positive Autocorrelation -> Momentum or Trend following
~~~
# Convert the daily data to weekly data
MSFT = MSFT.resample('W',how = 'last')
# Compute the percentage change of prices
returns = MSFT.pct_change()
# Compute and print the autocorrelation of returns
autocorrelation = returns['Adj Close'].autocorr()
print("The autocorrelation of weekly returns is %4.2f" %(autocorrelation))
~~~
#Autocorrelation Function
~~~
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
# Compute the acf array of HRB
acf_array = acf(HRB)
print(acf_array)
# Plot the acf function
plot_acf(HRB, alpha=1)
plt.show()
~~~ 
### negative means mean reverting
~~~
# Import the plot_acf module from statsmodels and sqrt from math
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from math import sqrt
# Compute and print the autocorrelation of MSFT weekly returns
autocorrelation = returns['Adj Close'].autocorr()
print("The autocorrelation of weekly MSFT returns is %4.2f" %(autocorrelation))
# Find the number of observations by taking the length of the returns DataFrame
nobs = len(returns)
# Compute the approximate confidence interval
conf = 1.96/sqrt(nobs)
print("The approximate confidence interval is +/- %4.2f" %(conf))
~~~

# White Noise (Can't forcast white noise)
#### stock return are useally white noise, for the white noise, we cannot forcast future observations based on the past autocrrelations at all lags are zero.
~~~
# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Simulate white noise returns
returns = np.random.normal(loc=0.02, scale=0.05, size=1000)

# Print out the mean and standard deviation of returns
mean = np.mean(returns)
std = np.std(returns)
print("The mean is %5.3f and the standard deviation is %5.3f" %(mean,std))

# Plot returns series
plt.plot(returns)
plt.show()

# Plot autocorrelation function of white noise returns
plot_acf(returns, lags=20)
plt.show()
~~~
# Random Walk
Whereas stock returns are often modelled as white noise, stock prices closely follow a random walk. In other words, today's price is yesterday's price plus some random noise.
~~~
# Generate 500 random steps with mean=0 and standard deviation=1
steps = np.random.normal(loc=0, scale=1, size=500)

# Set first element to 0 so that the first price will be the starting stock price
steps[0]=0

# Simulate stock prices, P with a starting price of 100
P = 100 + np.cumsum(steps)

# Plot the simulated stock prices
plt.plot(P)
plt.title("Simulated Random Walk")
plt.show()
~~~
Now you will make the noise multiplicative: you will add one to the random, normal changes to get a total return, and multiply that by the last price.
np.cumprod: Cumulate the product of the steps 

~~~
# Generate 500 random steps
steps = np.random.normal(loc=0.001, scale=0.01, size=500) + 1

# Set first element to 1
steps[0]=1

# Simulate the stock price, P, by taking the cumulative product
P = 100 * np.cumprod(steps)

# Plot the simulated stock prices
plt.plot(P)
plt.title("Simulated Random Walk with Drift")
~~~

With the ADF test, the "null hypothesis" (the hypothesis that we either reject or fail to reject) is that the series follows a random walk. Therefore, a low p-value (say less than 5%) means we can reject the null hypothesis that the series is a random walk.

~~~~
# Import the adfuller module from statsmodels
from statsmodels.tsa.stattools import adfuller

# Run the ADF test on the price series and print out the results
results = adfuller(AMZN['Adj Close'])
print(results)

# Just print out the p-value
print('The p-value of the test on prices is: ' + str(results[1]))
~~~


If we want to check whether a stock return(% change in price) follows Random Walk, we should run Augmented Dickey-Fuller test. 

~~~
# Import the adfuller module from statsmodels
from statsmodels.tsa.stattools import adfuller

# Create a DataFrame of AMZN returns
AMZN_ret = AMZN.pct_change()

# Eliminate the NaN in the first row of returns
AMZN_ret = AMZN_ret.dropna()

# Run the ADF test on the return series and print out the p-value
results = adfuller(AMZN_ret['Adj Close'])
print('The p-value of the test on returns is: ' + str(results[1]))
~~~

## Seasonal Adjustment During Tax Season
~~~
#Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Seasonally adjust quarterly earnings
HRBsa = HRB.diff(4)

# Print the first 10 rows of the seasonally adjusted series
print(HRBsa.head(10))

# Drop the NaN data in the first three three rows
HRBsa = HRBsa.dropna()

# Plot the autocorrelation function of the seasonally adjusted series
plot_acf(HRBsa)
plt.show()
~~~

# Describe AR Model (ArmaProcess)
## Simulate AR(1) Time Series
~~~
# import the module for simulating data
from statsmodels.tsa.arima_process import ArmaProcess

# Plot 1: AR parameter = +0.9
plt.subplot(2,1,1)
ar1 = np.array([1, -0.9])
ma1 = np.array([1])
AR_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = AR_object1.generate_sample(nsample=1000)
plt.plot(simulated_data_1)

# Plot 2: AR parameter = -0.9
plt.subplot(2,1,2)
ar2 = np.array([1, 0.9])
ma2 = np.array([1])
AR_object2 = ArmaProc
~~~


# Compare the ACF for Several AR Time Series
The autocorrelation function decays exponentially for an AR time series at a rate of the AR parameter. For example, if the AR parameter, ϕ=+0.9ϕ=+0.9, the first-lag autocorrelation will be 0.9, the second-lag will be (0.9)2=0.81(0.9)2=0.81, the third-lag will be (0.9)3=0.729(0.9)3=0.729, etc. A smaller AR parameter will have a steeper decay, and for a negative AR parameter, say -0.9, the decay will flip signs, so the first-lag autocorrelation will be -0.9, the second-lag will be (−0.9)2=0.81(−0.9)2=0.81, the third-lag will be (−0.9)3=−0.729(−0.9)3=−0.729, etc.
### Compare the ACF for Several AR Time Series
~~~
# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Plot 1: AR parameter = +0.9
plot_acf(simulated_data_1, alpha=1, lags=20)
plt.show()

# Plot 2: AR parameter = -0.9
plot_acf(simulated_data_2, alpha=1, lags=20)
plt.show()
~~~


# Plot 3: AR parameter = +0.3
plot_acf(simulated_data_3, alpha=1, lags=20)
plt.show()
~~~

## Estimating an AR Model
~~~
# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

# Fit an AR(1) model to the first simulated data
mod = ARMA(simulated_data_1, order=(1,0))
res = mod.fit()

# Print out summary information on the fit
print(res.summary())

# Print out the estimate for the constant and for phi
print("When the true phi=0.9, the estimate of phi (and the constant) are:")
print(res.params)
~~~

## Forecasting with an AR Model
~~~
# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

# Forecast the first AR(1) model
mod = ARMA(simulated_data_1, order=(1,0))
res = mod.fit()
res.plot_predict(start=990, end=1010)
plt.show()
~~~
## Forecasting intrest rate
~~~
# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

# Forecast interest rates using an AR(1) model
mod = ARMA(interest_rate_data, order=(1,0))
res = mod.fit()

# Plot the original series and the forecasted series
res.plot_predict(start=0,end ='2022')
plt.legend(fontsize=8)
plt.show()
~~~
# Chosing the Right Model
Identifying the order of an AR Model
The order of an AR(p) model will usually be unknown
Two techniques to determine order
-- Partial Autocorrelation Function (partial ACF)
-- Information criteria
