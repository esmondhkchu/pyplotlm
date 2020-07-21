# pyplotlm - R style linear regression diagnostic plots for sklearn

## Install
```bash
pip install pyplotlm
```

## Introduction
A replica of the R style `plot.lm` for Python sklearn. <br>
There are six plots avaiable:
1. Residuals vs Fitted
2. Normal Q-Q
3. Scale-Location
4. Cook's Distance
5. Residuals vs Leverage
6. Cook's Distance vs Leverage

## Usage
In R environment, we can fit a linear model and generate diagnostic plots by doing the following: <br>

```R
fit = lm(y ~ ., data=data)
par(mfrow=c(2,2))
plot(fit)
```

The results are as follow: <br>
![](https://github.com/esmondhkchu/pyplotlm/blob/dev/graph/R_plot.png) <br>

<br>
This package provides an analogy for sklearn linear_model object to return a similar result: <br>

```python
from sklearn import linear_model
import matplotlib.pyplot as plt

from pyplotlm import *

reg = linear_model.LinearRegression()
reg.fit(X, y)

plt.figure(figsize=(20,15))
PyPlotLm(reg, X, y).plot()
plt.show()
```

This will produce the same set of diagnostic plots: <br>
![](https://github.com/esmondhkchu/pyplotlm/blob/dev/graph/python_plot.png) <br>

## References:
1. Regression Deletion Diagnostics (R)
https://stat.ethz.ch/R-manual/R-devel/library/stats/html/influence.measures.html <br>
https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/lm <br>
https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/plot.lm <br>

2. Residuals and Influence in Regression
https://conservancy.umn.edu/handle/11299/37076 <br>
https://en.wikipedia.org/wiki/Studentized_residual <br>

3. Cook's Distance
https://en.wikipedia.org/wiki/Cook%27s_distance <br>
