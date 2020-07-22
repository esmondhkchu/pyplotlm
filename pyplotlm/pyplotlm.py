from .tools import *
from .influence import *
from .quantile import *

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

class PyPlotLm:
    def __init__(self, reg, X, y, intercept=False):
        """ plot a sklearn linear regression model, analogy to R plot(lm())
            there are six plots avaiable:
            1. Residuals vs Fitted
            2. Normal Q-Q
            3. Scale-Location
            4. Cook's Distance
            5. Residuals vs Leverage
            6. Cook's Distance vs Leverage

        Parameters: reg (sklearn.linear_model) - a fitted sklearn.linear_model object
                    X (nd-array) - the design matrix
                    y (array) - the response
                    intercept (boo) - if the X data has intercept or not
                                      optional, default is False

        Arributes: fitted_values (array) - fitted values, aka y_hat
                   residuals (array) - raw residuals, i.e y - y_hat
                   resid_max_3 (array) - top max 3 of raw residuals

                   h (array) - leverage
                   p (int) - total numbers of features, including intercept
                   n (int) - total numbers of observations

                   standard_residuals (array) - internally studentized residuals
                   root_standard_residuals (array) - square root of the absolute values of the
                                                     internally studentized residuals

                   theo_quantiles (array) - theorectical quantiles

                   cooks (array) - Cook's Distance
                   cooks_max_3 (array) - top max 3 of Cook's Distance

        References:
        1. Regression Deletion Diagnostics (R)
        https://stat.ethz.ch/R-manual/R-devel/library/stats/html/influence.measures.html
        https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/lm
        https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/plot.lm

        2. Residuals and Influence in Regression
        https://conservancy.umn.edu/handle/11299/37076
        https://en.wikipedia.org/wiki/Studentized_residual

        3. Cook's Distance
        https://en.wikipedia.org/wiki/Cook%27s_distance
        """

        if not isinstance(X, np.ndarray):
            raise TypeError('input design matrix must be a numpy array object')

        if X.shape[0] != len(y):
            raise DimensionError('X dimension must match with y dimension')

        self.reg = reg
        self.X = X
        self.y = y

        # fitted values from model
        self.fitted_values = self.reg.predict(self.X)
        # raw residuals from model
        self.residuals = self.y - self.fitted_values

        # top 3 max residuals
        self.resid_max_3 = np.argsort(abs(self.residuals))[::-1][:3]

        # leverage
        self.h = leverage(self.X, intercept)

        # studentized residuals
        if intercept:
            self.p = self.X.shape[-1]
        else:
            self.p = self.X.shape[-1] + 1

        self.n = self.X.shape[0]
        self.standard_residuals = internally_studentized(self.residuals, self.h, self.p, self.n)

        # square root of absolute studentized residuals
        self.root_standard_residuals = np.sqrt(abs(self.standard_residuals))

        # theorectical quantiles
        self.theo_quantiles = theorectical_quantiles(self.standard_residuals)

        # Cook's Distance
        self.cooks = cooks_distance(self.standard_residuals, self.h, self.p)
        self.cooks_max_3 = np.argsort(self.cooks)[::-1][:3]

    def plot(self, which=None, size=(12,10)):
        """ by default this method plots the most common 4 plots, which is 1, 2, 3 and 5
            i.e.
            1. Residuals vs Fitted
            2. Normal Q-Q
            3. Scale-Location
            5. Residuals vs Leverage

        4. Cook's Distance and 6. Cook's Distance vs Leverage plot aren't as common, so will exclude from default
        But, we can create these plot by using the 'which' parameters

        Parameters: which (int) - by default, it will plot the most common 4 plots, if which is specified, it will plot plot the specific plot
                    size (tuple) - the size for the 2x2 default plots
        """
        if which is not None:
            if which == 1:
                self.residuals_fitted()
            elif which == 2:
                self.normal_qq()
            elif which == 3:
                self.scale_location()
            elif which == 4:
                self.cooks_distance()
            elif which == 5:
                self.residual_leverage()
            elif which == 6:
                self.cooks_leverage()

        else:
            plt.figure(figsize=size)
            plt.subplots_adjust(hspace=0.3)

            plt.subplot(221)
            self.residuals_fitted()

            plt.subplot(222)
            self.normal_qq()

            plt.subplot(223)
            self.scale_location()

            plt.subplot(224)
            self.residual_leverage()

    def residuals_fitted(self):
        """ plot 1. Residuals vs Fitted
        """
        sns.residplot(self.fitted_values, self.residuals,
              lowess=True,
              scatter_kws={'alpha': 0.5},
              line_kws={'color': 'red', 'lw': 1})

        for i in self.resid_max_3:
            plt.annotate(i, xy=(self.fitted_values[i], self.residuals[i]))

        plt.title('Residuals vs Fitted', size=20)
        plt.xlabel('Fitted values', size=20)
        plt.ylabel('Residuals', size=20)

    def normal_qq(self):
        """ plot 2. Normal Q-Q
        """
        sns.regplot(self.theo_quantiles, sorted(self.standard_residuals),
            lowess=True,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'lw': 1})

        pos_idx = 0
        neg_idx = 0
        for i in self.resid_max_3:
            if self.standard_residuals[i] < 0:
                plt.annotate(i, xy=[self.theo_quantiles[neg_idx], sorted(self.standard_residuals)[neg_idx]])
                neg_idx += 1
            else:
                pos_idx += -1
                plt.annotate(i, xy=[self.theo_quantiles[pos_idx], sorted(self.standard_residuals)[pos_idx]])

                plt.title('Normal Q-Q', size=20)
                plt.xlabel('Theoretical Quantiles', size=20)
                plt.ylabel('Standardized residuals', size=20)

    def scale_location(self):
        """ plot 3. Scale-Location
        """
        sns.regplot(self.fitted_values, self.root_standard_residuals,
            lowess=True,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'lw': 1})

        for i in self.resid_max_3:
            plt.annotate(i, xy=[self.fitted_values[i], self.root_standard_residuals[i]])

        plt.title('Scale-Location', size=20)
        plt.xlabel('Fitted values', size=20)
        plt.ylabel('$\sqrt{|Standardized residuals|}$', size=20)

    def cooks_distance(self):
        """ plot 4. Cook's Distance
        """
        plt.bar(range(len(self.cooks)), self.cooks)

        # cook's distance max 3 annotation
        for i in self.cooks_max_3:
            plt.annotate(i, xy=[i, self.cooks[i]])

        plt.title("Cook's Distance", size=20)
        plt.xlabel('Obs. number', size=20)
        plt.ylabel("Cook's Distance", size=20)

    def residual_leverage(self):
        """ plot 5. Residuals vs Leverage
        """
        min_y = min(self.standard_residuals) + min(self.standard_residuals)*0.15
        max_y = max(self.standard_residuals) + max(self.standard_residuals)*0.15

        # define cook's distance components
        cooks_x = np.linspace(min(self.h), max(self.h), 50)
        cooks_y_1 = np.sqrt((self.p*(1-cooks_x)) / cooks_x)

        cooks_y_05_ = np.sqrt(0.5*(self.p*(1-cooks_x)) / cooks_x)

        cooks_y_1_neg = -np.sqrt((self.p*(1-cooks_x)) / cooks_x)
        cooks_y_05_neg = -np.sqrt(0.5*(self.p*(1-cooks_x)) / cooks_x)

        # main plot
        sns.regplot(self.h, self.standard_residuals,
            lowess=True,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'lw': 1})

        for i in self.cooks_max_3:
            plt.annotate(i, xy=[self.h[i], self.standard_residuals[i]])

        # plot cook's distance
        if any(cooks_y_1 < max_y):
            plt.plot(cooks_x, cooks_y_1, ls = ':', color = 'r', label="Cooks's Distance")

            for i in [cooks_y_1[-1], cooks_y_1_neg[-1]]:
                plt.annotate(1, xy=[cooks_x[-1], i], color='red')

            plt.legend(frameon=False)

        if any(cooks_y_05_ < max_y):
            plt.plot(cooks_x, cooks_y_05_, ls = ':', color = 'r')

            for i in [cooks_y_05_[-1], cooks_y_05_neg[-1]]:
                plt.annotate(0.5, xy=[cooks_x[-1], i], color='red')

        if any(cooks_y_1_neg > min_y):
            plt.plot(cooks_x, cooks_y_1_neg, ls = ':', color = 'r')

            for i in [cooks_y_1_neg[-1], cooks_y_05_neg[-1]]:
                plt.annotate(1, xy=[cooks_x[-1], i], color='red')

        if any(cooks_y_05_neg > min_y):
            plt.plot(cooks_x, cooks_y_05_neg, ls = ':', color = 'r')

            for i in [cooks_y_05_neg[-1], cooks_y_05_neg[-1]]:
                plt.annotate(1, xy=[cooks_x[-1], i], color='red')

        plt.ylim(min_y, max_y)

        plt.title('Residuals vs Leverage', size=20)
        plt.xlabel('Leverage', size=20)
        plt.ylabel('Standardized residuals', size=20)

    def cooks_leverage(self):
        """ plot 6. Cook's Distance vs Leverage
        """
        sns.regplot(self.h / (1-self.h), self.cooks,
            lowess=True,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'lw': 1})

        for i in self.cooks_max_3:
            plt.annotate(i, xy=[self.h[i], self.cooks[i]])

        plt.title("Cook's dist vs Leverage $h_{ii}/(1-h_{ii})$", size=20)
        plt.xlabel('Leverage $h_{ii}$', size=20)
        plt.ylabel("Cook's distance", size=20)
