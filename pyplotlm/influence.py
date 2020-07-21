import numpy as np
from numpy.linalg import *

def leverage(X, intercept=False):
    """ compute leverage of a design matrix

    Parameters: X (nd-array) - the design matrix
                intercept (boo) - if the design matrix input included intercept or not
                                  optional, default is False

    Returns: (array) - the leverage
    """
    if intercept:
        big_X = X
    else:
        big_X = np.concatenate([np.ones(X.shape[0]).reshape(-1,1), X], 1)

    # hat matrix
    # the orthogonal projection onto the column space of X
    H = np.dot(np.dot(big_X, np.linalg.inv(np.dot(big_X.T, big_X))), big_X.T)
    # diagonal of hat matrix, aka, leverage
    H_diag = H.diagonal()

    return H_diag

def internally_studentized(residuals, h, p, n):
    """ compute the Internally Studentized Residuals

    Parameters: residual (array-like) - the raw residuals
                h (array-like) - leverae of the design matrix
                p (int) - coefficients dimensions
                n (int) - total numbers of observations

    Returns: (array) - the studentized residual
    """
    sigma = np.sqrt((1/(n-p))*np.sum(residuals**2))
    standardized_residuals = residuals / (sigma*np.sqrt(1-h))
    return standardized_residuals

def cooks_distance(standard_residuals, h, p):
    """ compute cook's distance

    Parameters: standard_residuals (array) - studentized residauls
                h (array) - leverage
                p (int) - coefficients dimensions

    Returns: (array) - an array of Cook's distance
    """
    cooks = (h / (1-h)) * (standard_residuals**2 / p)
    return cooks
