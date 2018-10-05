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
