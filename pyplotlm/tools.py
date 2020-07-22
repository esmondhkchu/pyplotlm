import numpy as np
import matplotlib.pyplot as plt

class Error(Exception):
    """ base class
    """
    pass

class DimensionError(Error):
    """ raise when dimension mismatch
    """
    pass

def abline(intercept, slope, x_min=0, x_max=10, marker=':', color='black'):
    """Plot a line from slope and intercept
    """
    x_vals = np.linspace(x_min, x_max)
    y_vals = x_vals*slope + intercept
    plt.plot(x_vals, y_vals, marker, color=color)
