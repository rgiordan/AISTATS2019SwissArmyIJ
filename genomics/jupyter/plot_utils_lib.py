# Helper functions for the Jupyter notebooks.

import numpy as np
import matplotlib.pyplot as plt

def PlotPredictionLine(timepoints, regs, pred, n, this_plot):
    this_plot.plot(timepoints, pred[n, :] + np.mean(regs.y[n]),
                   color='green', linewidth=3)

def PlotRegressionLine(timepoints, regs, reg_params, n,
                       num_draws=30, this_plot=None):
    if this_plot is None:
        _, this_plot = plt.subplots(1, 1, figsize=(15,8))
    beta_mean = reg_params['beta_mean'][n, :]
    beta_cov = np.linalg.inv(reg_params['beta_info'][n, :, :])
    this_plot.plot(timepoints, regs.y[n, :], '+', color = 'blue');
    this_plot.plot(timepoints, regs.x @ beta_mean, color = 'red');
    this_plot.set_ylabel('gene expression')
    this_plot.set_xlabel('time')
    this_plot.set_title('gene number {}'.format(n))

    # draw from the variational distribution, to plot uncertainties
    for j in range(num_draws):
        beta_draw = np.random.multivariate_normal(beta_mean, beta_cov)
        this_plot.plot(timepoints, regs.x @ beta_draw,
                       color = 'red', alpha = 0.08);
