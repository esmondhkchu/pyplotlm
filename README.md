# pyplotlm - R style linear regression diagnostic plots for sklearn
This package is a reproduction of the `plot.lm` function in R but for a python environment and is meant to support the sklearn by adding diagnostic plots for linear regression. <br>
In the R environment, we can fit a linear model and generate diagnostic plots by doing the following: <br>
```R
fit = lm(y ~ ., data=data)
par(mfrow=c(2,2))
plot(fit)
```
![](https://github.com/esmondhkchu/pyplotlm/blob/dev/graph/R_plot.png) <br>
The goal of this package is to make the process of producing diagnostic plots as simple as it is in R.

## Install
```bash
pip install pyplotlm
```

## Introduction
There are two core functionalities:

A. style summary of regression report <br>
B. six plots avaiable: <br>
    1. Residuals vs Fitted
    2. Normal Q-Q
    3. Scale-Location
    4. Cook's Distance
    5. Residuals vs Leverage
    6. Cook's Distance vs Leverage

## Usage
Below is how you would produce the diagnostic plots in Python:
```python
>>> from sklearn.datasets import load_diabetes
>>> from sklearn.linear_model import LinearRegression
>>> from pyplotlm import *

>>> X, y = load_diabetes(return_X_y=True)

>>> reg = LinearRegression().fit(X, y)

>>> obj = PyPlotLm(reg, X, y, intercept=False)
>>> obj.summary()
Residuals:
       Min        1Q   Median       3Q       Max
 -155.8290  -38.5339  -0.2269  37.8061  151.3550

Coefficients:
               Estimate Std. Error     t value   Pr(>|t|)
(Intercept)   1.521e+02  2.576e+00   5.906e+01        0.0  ***
X0           -1.001e+01  5.975e+01  -1.676e-01      0.867
X1           -2.398e+02  6.122e+01  -3.917e+00  0.0001041  ***
X2            5.198e+02  6.653e+01   7.813e+00  4.308e-14  ***
X3            3.244e+02  6.542e+01   4.958e+00  1.024e-06  ***
X4           -7.922e+02  4.167e+02  -1.901e+00    0.05795  .
X5            4.767e+02  3.390e+02   1.406e+00     0.1604
X6            1.010e+02  2.125e+02   4.754e-01     0.6347
X7            1.771e+02  1.615e+02   1.097e+00     0.2735
X8            7.513e+02  1.719e+02   4.370e+00  1.556e-05  ***
X9            6.763e+01  6.598e+01   1.025e+00      0.306
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 54.154 on 431 degrees of freedom
Multiple R-squared: 0.5177,     Adjusted R-squared: 0.5066
F-statistic: 4.6e+01 on 10 and 431 DF,  p-value: 1.11e-16

>>> obj.plot()
>>> plt.show()
```
This will produce the same set of diagnostic plots: <br>
![](https://github.com/esmondhkchu/pyplotlm/blob/dev/graph/python_plot.png) <br>

## References:
1. Regression Deletion Diagnostics (R) <br>
https://stat.ethz.ch/R-manual/R-devel/library/stats/html/influence.measures.html <br>
https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/lm <br>
https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/plot.lm <br>

2. Residuals and Influence in Regression <br>
https://conservancy.umn.edu/handle/11299/37076 <br>
https://en.wikipedia.org/wiki/Studentized_residual <br>

3. Cook's Distance <br>
https://en.wikipedia.org/wiki/Cook%27s_distance <br>
