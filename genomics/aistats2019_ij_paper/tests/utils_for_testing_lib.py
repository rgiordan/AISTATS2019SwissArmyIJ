import numpy as np
from numpy.testing import assert_array_almost_equal

import aistats2019_ij_paper.regression_lib as reg_lib
import aistats2019_ij_paper.regression_mixture_lib as rm_lib
import aistats2019_ij_paper.saving_gmm_utils


def get_sample_timepoints(reps=3):
    """Get timepoints that mimic the spacing in the real data.
    """
    timepoints = np.array([
        0.,   3.,   6.,   9.,  12.,  18.,  24.,
        30.,  36.,  48.,  60.,  72., 120., 168.])
    return np.repeat(timepoints, reps)


def assert_gmm_equal(testcase, gmm, new_gmm):
    assert_array_almost_equal(
        gmm.transform_mat, new_gmm.transform_mat)
    assert_array_almost_equal(
        gmm.unrotate_transform_mat, new_gmm.unrotate_transform_mat)
    testcase.assertEqual(gmm.num_obs, new_gmm.num_obs)
    testcase.assertEqual(gmm.num_components, new_gmm.num_components)
    testcase.assertEqual(gmm.gmm_params_pattern, new_gmm.gmm_params_pattern)
    testcase.assertEqual(gmm.inflate_coef_cov, new_gmm.inflate_coef_cov)
    testcase.assertEqual(gmm.cov_regularization, new_gmm.cov_regularization)
    testcase.assertEqual(gmm.obs_dim, new_gmm.obs_dim)
    for prior_param, v in gmm.prior_params.items():
        assert_array_almost_equal(
            v, new_gmm.prior_params[prior_param])

    hess_dim = gmm.gmm_params_pattern.flat_length(free=True)
    assert_array_almost_equal(
        gmm.get_kl_conditioned.get_preconditioner(hess_dim),
        new_gmm.get_kl_conditioned.get_preconditioner(hess_dim))


def assert_regressions_equal(testcase, regs, new_regs):
    assert_array_almost_equal(regs.x, new_regs.x)
    assert_array_almost_equal(regs.y, new_regs.y)
    testcase.assertEqual(
        regs._reverse_jac_order, new_regs._reverse_jac_order)


def assert_fit_derivs_equal(testcase, fit_derivs, new_fit_derivs):
    assert_array_almost_equal(
        np.array(fit_derivs.full_hess.todense()),
        np.array(new_fit_derivs.full_hess.todense()))
    assert_array_almost_equal(
        np.array(fit_derivs.t_jac.todense()),
        np.array(new_fit_derivs.t_jac.todense()))
    assert_array_almost_equal(
        fit_derivs.comb_params_free,
        new_fit_derivs.comb_params_free)
    assert_array_almost_equal(
        fit_derivs.reg_params_free,
        new_fit_derivs.reg_params_free)
    assert_array_almost_equal(
        fit_derivs.gmm_params_free,
        new_fit_derivs.gmm_params_free)
    testcase.assertEqual(fit_derivs.comb_params_pattern,
                         new_fit_derivs.comb_params_pattern)


def simulate_data(x, y_scale, true_k, n_obs,
                  beta_prior_mean, beta_prior_cov,
                  log_shift_mean, log_shift_scale):
    """Simulate genomics time series data.

    The model is

    .. math ::
        y_g \sim \mathcal{N}(x \\beta_{k_g} + a_g, \\sigma_y^2)
        \\k_g \sim \\mathrm{Categorical}(\\pi_1, ..., \\pi_K)
        \\beta_k \\sim \mathcal{N}(\\mu_{\\beta}, \\Sigma_{\\beta})
        \\log(a_g) \\sim \mathcal{N}(\\mu_a, \\sigma_a^2)

    Here,
    ``y_scale`` = :math:`\\sigma_y`,
    ``beta_prior_mean`` = :math:`\\mu_{\\beta}`,
    ``beta_prior_cov`` = :math:`\\Sigma_{\\beta}`,
    ``log_shift_mean`` = :math:`\\mu_a`,
    ``log_shift_scale`` = :math:`\\sigma_a`, and
    ``true_k`` = :math:`K`.
    """
    d = np.shape(x)[0] # number of time points
    r = np.shape(x)[1] # number of basis vectors

    # Mixing proportions
    true_pi = np.ones(true_k)
    true_pi = true_pi / true_k

    # draw group indicators
    true_z_ind = np.random.choice(range(true_k), p = true_pi, size = n_obs)
    true_z = np.zeros((n_obs, true_k))
    for i in range(n_obs):
        true_z[i, true_z_ind[i]] = 1.0

    # Draw the shifts.
    true_a = np.exp(np.random.normal(
        loc=log_shift_mean, scale=log_shift_scale, size=n_obs))

    # Draw the regression coefficients.
    true_beta = np.array([
        np.random.multivariate_normal(beta_prior_mean, beta_prior_cov) \
        for k in range(true_k) ]).T

    # draw observations
    y = np.array([ np.random.multivariate_normal(
                       x @ true_beta[:, true_z_ind[n]] + true_a[n],
                       (y_scale ** 2) * np.eye(d)) \
               for n in range(n_obs) ])

    return y, true_z_ind, true_z, true_beta, true_a
