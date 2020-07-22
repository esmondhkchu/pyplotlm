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
There are six plots available:
1. Residuals vs Fitted
2. Normal Q-Q
3. Scale-Location
4. Cook's Distance
5. Residuals vs Leverage
6. Cook's Distance vs leverage / (1 - leverage)

## Usage
Below is how you would produce the diagnostic plots in Python:
```python
from sklearn import linear_model
import matplotlib.pyplot as plt

from pyplotlm import *

reg = linear_model.LinearRegression()
reg.fit(X, y)

PyPlotLm(reg, X, y).plot()
plt.show()
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
