#### Imports ####

import math

import statistics

import numpy as np

import scipy.stats

import pandas as pd

import matplotlib.pyplot as plt



#### Descriptive statistics ####

## we can start by creating arbitrary data

x = [8.0, 1, 2.5, 4, 28.0] ## dataset we define as x 

x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0] ## dataset that matches x but has a NaN value
## to insert a NaN value into the dataset directly, we can use math.nan, float('nan'), or np.nan
## for example, we could've defined x_with_nan like this:

x_with_nan = [8.0, 1, 2.5, np.nan, 4, 28.0]

math.isnan(x_with_nan[3]), np.isnan(x_with_nan[3]) ## we can also write this function to prove their equality
## they both are able to correctly identify the element as NaN when prompted with position

## it should be noted that when directly comparing two NaN values, however, it will always return false
## i.e.

math.nan == math.nan ## output: False

## let's take a look
x

x_with_nan 

## now let's create a numpy array and pandas series corresponding to both x and x_with_nan
## y for np and z for pd

y, y_with_nan = np.array(x), np.array(x_with_nan)

z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)

## let's take a look

y 

y_with_nan

z

z_with_nan

## note: pd also allows the user to specify a label for each element in z and z_with_nan




#### Measures of Central Tendency ####

## Arithmetic Mean ##

## python has built in functionality for calculating mean, through sum() and len()

mean_ = sum(x) / len(x)

mean_

## you can also use built in stats functionality

mean_ = statistics.mean(x)

mean_

mean_ = statistics.fmean(x)

mean_

## the difference between mean() and fmean() is that fmean() will always return a floating-point number and is meant to be faster
## mean and fmean will always return nan when there is a nan value in the set
## this is because sum returns nan when nan is present

mean_ = statistics.mean(x_with_nan)

mean_

mean_ = statistics.fmean(x_with_nan)

mean_

## numpy also has a mean functionality 

mean_ = np.mean(y)

mean_

mean_ = y.mean() ## alternatively, you can write it like this

mean_

## np.mean() will also return nan when nan values are present

np.mean(y_with_nan)

y_with_nan.mean()

## np has .nanmean(), which is useful because it ignores all nan values and gives the mean as if there are no nan values

np.nanmean(y_with_nan)

## pandas has mean functionality as well, unsurprisingly

mean_ = z.mean()

mean_

z_with_nan.mean() ## pandas mean skips nan values by default
## to change this, modify parameter skipna when calling pd.mean()


## Weighted Mean ##

## to calculate weighted mean, one can multiply the frequency of a value by the value and sigma the whole thing

## in base python, a user can utilize sum() with either range() or zip() to calculate weighted mean

x = [8.0, 1, 2.5, 4, 28.0]

w = [0.1, 0.2, 0.3, 0.25, 0.15]

wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)

wmean

wmean = sum(x_ * w_ for (x_, w_) in zip(x, w)) / sum(w)

wmean

## numpy is better suited for weighted mean for larger datasets

y, z, w = np.array(x), pd.Series(x), np.array(w)

wmean = np.average(y, weights=w)

wmean

wmean = np.average(z, weights=w)

wmean

## or you can simply use operations 

(w * y).sum() / w.sum()

## nan values will mess with numpys wmean, as expected

w = np.array([0.1, 0.2, 0.3, 0.0, 0.2, 0.1])

(w * y_with_nan).sum() / w.sum()

np.average(y_with_nan, weights=w)

np.average(z_with_nan, weights=w)


## Harmonic mean ##

## pure python 

hmean = len(x) / sum(1 / item for item in x)

hmean

## statistics hmean

hmean = statistics.harmonic_mean(x)

hmean

## special cases 

statistics.harmonic_mean(x_with_nan) ## any nan value will cause output nan

statistics.harmonic_mean([1, 0, 2]) ## any 0s will cause output 0

statistics.harmonic_mean([1, 2, -2]) ## any negatives will cause output error

## scipy stats hmean

scipy.stats.hmean(y)

scipy.stats.hmean(z)


## Geometric Mean ##

## pure python 

gmean = 1
for item in x:
     gmean *= item

gmean **= 1 / len(x)

gmean

## statistics geometric mean

gmean = statistics.geometric_mean(x)

gmean

gmean = statistics.geometric_mean(x_with_nan) ## this will return nan, like most other mean functions with nan as input

gmean

## scipy stats g mean

scipy.stats.gmean(y)

scipy.stats.gmean(z)

## nan inputs will output nan, any 0s will return 0, any negative numbers will give nan


## Median ##

## pure python example 

n = len(x)

if n % 2:
    median_ = sorted(x)[round(0.5*(n-1))]
else:
    x_ord, index = sorted(x), round(0.5 * n)
    median_ = 0.5 * (x_ord[index-1] + x_ord[index])

median_

## statistics median 

median_ = statistics.median(x)

median_

median_ = statistics.median(x[:-1])

median_

statistics.median_low(x[:-1])

statistics.median_high(x[:-1])

## median_low and _high can be used when there is an even number of values. low returns the lower of the two middle values and high returns the higher of the two

statistics.median(x_with_nan)

statistics.median_low(x_with_nan)

statistics.median_high(x_with_nan) ## it's important to note that statistics median doesn't return nan when nan values are involved

## numpy median

median_ = np.median(y)

median_

median_ = np.median(y[:-1])

median_

## np.median() will return nan if there is a nan value
## to get around this, one can use np.nanmedian()

np.nanmedian(y_with_nan)

np.nanmedian(y_with_nan[:-1])

## pandas median

z.median()

z_with_nan.median() ## ignores nan values by default, can change through parameter skipna


## Mode ##

## pure python example

u = [2, 3, 2, 8, 12]

mode_ = max((u.count(item), item) for item in set(u))[1] ## u.count() gets the number of occurences of each item in u

mode_

## statistics mode and multimode

mode_ = statistics.mode(u)

mode_

mode_ = statistics.multimode(u)

mode_

## multimode can handle sets with multiple modes and will return a list of the modes, whereas mode will return error if there are multiple modes

v = [12, 15, 12, 15, 21, 15, 12]

statistics.mode(v)  ## error

statistics.multimode(v)

## statistics mode and multimode can handle nan values as well

statistics.mode([2, math.nan, 2])

statistics.multimode([2, math.nan, 2])

statistics.mode([2, math.nan, 0, math.nan, 5])

statistics.multimode([2, math.nan, 0, math.nan, 5])

## scipy mode

u, v = np.array(u), np.array(v)

mode_ = scipy.stats.mode(u)

mode_

mode_ = scipy.stats.mode(v)

mode_

## if there are multiple modes, the smallest is returned

## numpy mode

mode_.mode ## returns mode

mode_.count ## returns number of occurences of mode


## pandas mode

u, v, w = pd.Series(u), pd.Series(v), pd.Series([2, 2, math.nan])

u.mode()

v.mode()

w.mode()

## handles multimodal scenarios and ignores nan by default
## to account for nan values, input parameter dropna=False



#### Measures of Variability ####

## Variance ##

## pure python example

n = len(x)

mean_ = sum(x) / n

var_ = sum((item - mean_)**2 for item in x) / (n - 1)

var_

## statistics variance

var_ = statistics.variance(x)

var_

## this will return nan if there is a nan input


## numpy variance

var_ = np.var(y, ddof=1)

var_

var_ = y.var(ddof=1)

var_

## it's important to define delta degrees of freedom (ddof) as 1. this allows n-1 to be the denominator instead of n when calculating variance

## np var will return nan if nan values are in the data set

np.nanvar(y_with_nan, ddof=1) ## nanvar can circumvent this issue


## pandas variance

z.var(ddof=1)

z_with_nan.var(ddof=1) ## although pd var has ddof, its default value is 1, so it can be skipped

## it also skips nan values by default. to modify, change skipna parameter

## population variance is calculated by changing the denominator to 0 isntead of 1. this can be done by modifying ddof value or using statistics.pvariance() instead of statistics.variance()


## Standard Deviation ##

## pure python example

std_ = var_ ** 0.5

std_

## statistics std

std_ = statistics.stdev(x)

std_

## numpy std

np.std(y, ddof=1)

y.std(ddof=1)

np.std(y_with_nan, ddof=1)

y_with_nan.std(ddof=1)

np.nanstd(y_with_nan, ddof=1) ## it's important to specify ddof as 1 for std as well.
## .nanstd ignores nan values lest the output be nan for regular np.std

## pd std

z.std(ddof=1)

z_with_nan.std(ddof=1) ## as consistent with some of the other pd functions reviewed in this repo, ignores nan by default
## default ddof value is 1, and nan behavior can be changed with skipna parameter


## Population Standard Deviation ##

## change ddof to 0 in np or pd or use statistics.pstdev() instead of statistics.stdev()


## Skewness ##

## skewness can be calculated using mean and std

## pure python example

x = [8.0, 1, 2.5, 4, 28.0]

n = len(x)

mean_ = sum(x) / n

var_ = sum((item - mean_)**2 for item in x) / (n - 1)

std_ = var_ ** 0.5

skew_ = (sum((item - mean_)**3 for item in x)
          * n / ((n - 1) * (n - 2) * std_**3))

skew_

## the skew is postive, meaning there is a right-side tail

## scipy skewness

y, y_with_nan = np.array(x), np.array(x_with_nan)

scipy.stats.skew(y, bias=False)

scipy.stats.skew(y_with_nan, bias=False) ## bias set to False to enable corrections for statistical bias
## parameter nan_policy can be set to propagate, raise, or omit to change its behavior regarding nan values

## pd skewness

z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)

z.skew()

z_with_nan.skew()

## ignores nan by default, parameter skipna


## Percentiles ##

## statistics quantiles

x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]

statistics.quantiles(x, n=2)

statistics.quantiles(x, n=4, method='inclusive') ## in this example, 8.0 is the median of x, while 0.1 and 21.0 are the sample 25th and 75th percentiles, respectively. The parameter n defines the number of resulting equal-probability percentiles, and method determines how to calculate them

## numpy percentiles and quantiles

y = np.array(x)

np.percentile(y, 5) ## fifth percentile

np.percentile(y, 95) ## 95th percentile

np.percentile(y, [25, 50, 75]) ## can also calculate multiple percentiles at once, returning array

np.median(y) ## to check that our 50th percentile is correct

y_with_nan = np.insert(y, 2, np.nan)

y_with_nan

np.nanpercentile(y_with_nan, [25, 50, 75]) ## to ignore nan values

np.quantile(y, 0.05)

np.quantile(y, 0.95)

np.quantile(y, [0.25, 0.5, 0.75])

np.nanquantile(y_with_nan, [0.25, 0.5, 0.75]) ## np.quantile has similar functionality to .percentile but takes values between 0 and 1 instead of 0 and 100


## pandas quantiles

z, z_with_nan = pd.Series(y), pd.Series(y_with_nan)

z.quantile(0.05)

z.quantile(0.95)

z.quantile([0.25, 0.5, 0.75])

z_with_nan.quantile([0.25, 0.5, 0.75])


## Range ##

## pure python example 

y.max() - y.min()

z.max() - z.min()

z_with_nan.max() - z_with_nan.min() ## python's default max and min ignore nan

## np range

np.amax(y) - np.amin(y)

np.nanmax(y_with_nan) - np.nanmin(y_with_nan) ## to ignore nan values (although in this example it does not make a difference)

np.ptp(y)

np.ptp(z)

np.ptp(y_with_nan) ## returns nan if there is a nan value in the np array

np.ptp(z_with_nan) ## returns number if there is a nan value in a pd series

## interesting behavior


## Interquartile Range ##

quartiles = np.quantile(y, [0.25, 0.75])

quartiles[1] - quartiles[0]

quartiles = z.quantile([0.25, 0.75])

quartiles[0.75] - quartiles[0.25]


## Summary of Desc Statistics ##

## scipy describe

result = scipy.stats.describe(y, ddof=1, bias=False)

result ## awesome
## this function also takes parameter nanpolicy

result.nobs ## number of observations

result.minmax[0]  ## min tuple with the minimum and maximum values of your dataset

result.minmax[1]  ## max tuple with the minimum and maximum values of your dataset

result.mean

result.variance

result.skewness

result.kurtosis

## pd describe (better?) ##

result = z.describe() ## there is a optional parameter percentiles to modify the percentiles that show up in the results

result


## Measures of Correlation Between Pairs of Data ##

x = list(range(-10, 11))

y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]

x_, y_ = np.array(x), np.array(y)

x__, y__ = pd.Series(x_), pd.Series(y_)


## Covariance ##

## pure python example

n = len(x)

mean_x, mean_y = sum(x) / n, sum(y) / n

cov_xy = (sum((x[k] - mean_x) * (y[k] - mean_y) for k in range(n))
           / (n - 1))

cov_xy

## numpy covariance matrix

cov_matrix = np.cov(x_, y_) ## optional parameters bias default False, ddof default None

cov_matrix ## upper left is cov of x, and lower right is cov of y

x_.var(ddof=1)

y_.var(ddof=1) ## let's check our work

cov_xy = cov_matrix[0, 1]

cov_xy

cov_xy = cov_matrix[1, 0]

cov_xy ## the other two are the cov of x and y

## pandas covariance

cov_xy = x__.cov(y__)

cov_xy

cov_xy = y__.cov(x__) ## note the way that the function can be written

cov_xy


## Correlation Coefficient ##

## pure python example

var_x = sum((item - mean_x)**2 for item in x) / (n - 1)

var_y = sum((item - mean_y)**2 for item in y) / (n - 1)

std_x, std_y = var_x ** 0.5, var_y ** 0.5

r = cov_xy / (std_x * std_y)

r

## scipy stats pearsonr()

r, p = scipy.stats.pearsonr(x_, y_)

r ## returns correlation coefficient

p ## returns p value


## numpy correlation coefficient matrix

corr_matrix = np.corrcoef(x_, y_)

corr_matrix

r = corr_matrix[0, 1]

r

r = corr_matrix[1, 0]

r ## directly get the coefficient instead of returning matrix
## this can be applied to pure python and pearsonr() (as we saw)

scipy.stats.linregress(x_, y_) ## .linregress performs linear regression and returns results

## to isolate r

result = scipy.stats.linregress(x_, y_)

r = result.rvalue

r


## pandas correlation coefficient

r = x__.corr(y__)

r

r = y__.corr(x__)

r




#### Working with 2D Data ####

a = np.array([[1, 1, 1],
              [2, 3, 1],
              [4, 9, 2],
              [8, 27, 4],
              [16, 1, 1]])

a

## applying some statistics

np.mean(a)

a.mean()

np.median(a)

a.var(ddof=1)

## working with axes

## axis=None says to calculate the statistics across all data in the array. The examples above work like this. This behavior is often the default in NumPy.
## axis=0 says to calculate the statistics across all rows, that is, for each column of the array. This behavior is often the default for SciPy statistical functions.
## axis=1 says to calculate the statistics across all columns, that is, for each row of the array.

np.mean(a, axis=0)

a.mean(axis=0) ## for all columns

np.mean(a, axis=1)

a.mean(axis=1) ## for all rows

np.median(a, axis=0)

np.median(a, axis=1)

a.var(axis=0, ddof=1)

a.var(axis=1, ddof=1)

scipy.stats.gmean(a)  # default: axis=0

scipy.stats.gmean(a, axis=0)

scipy.stats.gmean(a, axis=1)

scipy.stats.gmean(a, axis=None) ## for entire dataset

scipy.stats.describe(a, axis=None, ddof=1, bias=False)

scipy.stats.describe(a, ddof=1, bias=False)  # default: axis=0

scipy.stats.describe(a, axis=1, ddof=1, bias=False) ## be careful with axis parameter when using describe with 2D data

result = scipy.stats.describe(a, axis=1, ddof=1, bias=False)

result.mean ## isolating a particular value with dot notation



## Data Frames ##

row_names = ['first', 'second', 'third', 'fourth', 'fifth']

col_names = ['A', 'B', 'C']

df = pd.DataFrame(a, index=row_names, columns=col_names)

df

df.mean()

df.var() ## by default, returns values for all columns

df.mean(axis=1)

df.var(axis=1) ## specifying axis=1 will return results for each row

df['A'] ## to isolate a single column of a df

df['A'].mean()

df['A'].var() ## after isolation, it is easy to get the results for just that column

df.values

df.to_numpy() ## sometimes it is appropriate to transfer the pd df to a numpy array

df.describe() ## again, by default, returning results for each column

df.describe().at['mean', 'A']

df.describe().at['50%', 'B'] ## accessing specific values




#### Visualizing Data ####

## matplotlib.pyplot is one of the most popular libraries to use for visualization
## it is already included in imports above

plt.style.use('ggplot')


## Boxplot ##

np.random.seed(seed=0)

x = np.random.randn(1000)

y = np.random.randn(100)

z = np.random.randn(10)

fig, ax = plt.subplots()

ax.boxplot((x, y, z), vert=False, showmeans=True, meanline=True,
           labels=('x', 'y', 'z'), patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'purple'},
           meanprops={'linewidth': 2, 'color': 'red'})

plt.show()

## .boxplot parameters
##x is your data.
## vert sets the plot orientation to horizontal when False. The default orientation is vertical.
## showmeans shows the mean of your data when True.
## meanline represents the mean as a line when True. The default representation is a point.
## labels: the labels of your data.
## patch_artist determines how to draw the graph.
## medianprops denotes the properties of the line representing the median.
## meanprops indicates the properties of the line or dot representing the mean.


## Histograms ##

hist, bin_edges = np.histogram(x, bins=10)

hist

bin_edges

fig, ax = plt.subplots()

ax.hist(x, bin_edges, cumulative=False)

ax.set_xlabel('x')

ax.set_ylabel('Frequency')

plt.show()

## changing cumulative to True

fig, ax = plt.subplots()

ax.hist(x, bin_edges, cumulative=True)

ax.set_xlabel('x')

ax.set_ylabel('Frequency')

plt.show()


## Pie Charts ##

x, y, z = 128, 256, 1024

fig, ax = plt.subplots()

ax.pie((x, y, z), labels=('x', 'y', 'z'), autopct='%1.1f%%')

plt.show()


## Bar Charts ##

x = np.arange(21)

y = np.random.randint(21, size=21)

err = np.random.randn(21)

fig, ax = plt.subplots()

ax.bar(x, y, yerr=err)

ax.set_xlabel('x')

ax.set_ylabel('y')

plt.show() ## barh() can be used for horizontal bars


## X-Y Plots ##

x = np.arange(21)

y = 5 + 2 * x + 2 * np.random.randn(21)

slope, intercept, r, *__ = scipy.stats.linregress(x, y)

line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'

fig, ax = plt.subplots()

ax.plot(x, y, linewidth=0, marker='s', label='Data points')

ax.plot(x, intercept + slope * x, label=line)

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.legend(facecolor='white')

plt.show()


## Heat Maps ##

matrix = np.cov(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()

## correlation coefficient heat map

matrix = np.corrcoef(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()