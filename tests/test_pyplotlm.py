import pytest
from pyplotlm import *

import pandas as pd

from sklearn import linear_model

# load data
data = pd.read_csv('boston.csv')
X = data[[str(i) for i in range(11) if i != 'y']].values
y = data['y'].values

# create a testing sklearn linear regression obj
reg = linear_model.LinearRegression()
reg.fit(X, y)

def test_X_is_ndarray():
    """ X has to be a numpy array object
        test if it properly raises a TypeError
    """
    with pytest.raises(TypeError):
        my_obj = PyPlotLm(reg, pd.DataFrame(X), y)

def test_X_y_shape_match():
    """ X dimension has to match with y length
        test if it properly raises a DimensionError
    """
    with pytest.raises(DimensionError):
        my_obj = PyPlotLm(reg, X, y[1:])
