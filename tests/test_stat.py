from pyplotlm import *

import numpy as np
import pandas as pd

from sklearn import linear_model

# load data
data = pd.read_csv('boston.csv')
X = data[[str(i) for i in range(11) if i != 'y']].values
y = data['y'].values

# create a testing sklearn linear regression obj
reg = linear_model.LinearRegression()
reg.fit(X, y)

# load calculated values from R

residuals_r = pd.read_csv('residuals.csv', header=None).values.reshape(-1)
leverage_r = pd.read_csv('leverage.csv', header=None).values.reshape(-1)
stand_resid_r = pd.read_csv('standard_residuals.csv', header=None).values.reshape(-1)
cooks_r = pd.read_csv('cooks.csv', header=None).values.reshape(-1)

# calculate from package
my_obj = PyPlotLm(reg, X, y)

residuals_py = my_obj.residuals
leverage_py = my_obj.h
stand_resid_py = my_obj.standard_residuals
cooks_py = my_obj.cooks

def test_residuals():
    """ test if residuals match with calculation from R
    """
    r_ver = np.array([round(i, 6) for i in residuals_r])
    py_ver = np.array([round(i, 6) for i in residuals_py])

    assert(all(r_ver == py_ver))

def test_leverage():
    """ test if leverage match with calculation from R
    """
    r_ver = np.array([round(i, 6) for i in leverage_r])
    py_ver = np.array([round(i, 6) for i in leverage_py])

    assert(all(r_ver == py_ver))

def test_studentized_residuals():
    """ test if studentized residuals match with calculation from R
    """
    r_ver = np.array([round(i, 6) for i in stand_resid_r])
    py_ver = np.array([round(i, 6) for i in stand_resid_py])

    assert(all(r_ver == py_ver))

def test_cooks_distance():
    """ test if Cook's Distance match with calculation from R
    """
    r_ver = np.array([round(i, 6) for i in cooks_r])
    py_ver = np.array([round(i, 6) for i in cooks_py])

    assert(all(r_ver == py_ver))
