#!/usr/bin/env python3

import autograd
import copy
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

import paragami

from aistats2019_ij_paper.tests import utils_for_testing_lib
from aistats2019_ij_paper.tests import generate_test_fit

from aistats2019_ij_paper import regression_lib as reg_lib
from aistats2019_ij_paper import regression_mixture_lib as rm_lib
from aistats2019_ij_paper import saving_gmm_utils

from aistats2019_ij_paper import spline_bases_lib
from aistats2019_ij_paper import sensitivity_lib as sens_lib


class TestMixture(unittest.TestCase):
    def test_regression_mixture(self):
        regs = generate_test_fit.load_test_regs()
        comb_params, metadata = generate_test_fit.load_test_fit()
        timepoints = metadata['timepoints']

        opt_reg_params = comb_params['reg']

        num_components = 5
        trans_obs_dim = regs.x.shape[1] - 1
        prior_params = \
            rm_lib.get_base_prior_params(trans_obs_dim, num_components)

        prior_params['probs_alpha'][:] = 1
        prior_params['centroid_prior_info'] = 1e-5 * np.eye(trans_obs_dim)

        gmm = rm_lib.GMM(num_components, prior_params, regs, opt_reg_params,
                         inflate_coef_cov=None,
                         cov_regularization=0.1)

        init_params = \
            rm_lib.kmeans_init(gmm.transformed_reg_params,
                               gmm.num_components, 5)
        init_params_flat = gmm.gmm_params_pattern.flatten(init_params, free=True)

        log_lik_by_nk, e_z = rm_lib.wrap_get_loglik_terms(
            init_params, gmm.transformed_reg_params)
        self.assertEqual((regs.num_obs, num_components), log_lik_by_nk.shape)
        self.assertEqual((regs.num_obs, num_components), e_z.shape)

        unique_timepoints = np.unique(timepoints)
        n_unique_timepoints = len(unique_timepoints)

        gmm_opt, opt_free_params = \
                gmm.optimize(init_params_flat, gtol=1e-2)

        gmm.initialize_preconditioner()
        gmm_opt, gmm_opt_x = gmm.optimize_fully(
            init_params_flat, verbose=True, gtol=1e-3)

        np.random.seed(12312)
        time_w = rm_lib.draw_time_weights(regs)
        rm_lib.refit_with_time_weights(
            gmm, regs, time_w, gmm_opt_x, verbose=False, gtol=1e-3)

    def test_e_z_update(self):
        # Test that the manual e_z update is at an optimum of the KL.
        np.random.seed(234324)
        regs = generate_test_fit.load_test_regs()
        comb_params, metadata = generate_test_fit.load_test_fit()
        gmm = generate_test_fit.load_test_gmm(regs, comb_params['reg'])

        pert = 0.01 * np.random.random(
            gmm.gmm_params_pattern.flat_length(free=True))
        gmm_params_free = gmm.gmm_params_pattern.flatten(
            comb_params['mix'], free=True)
        gmm_params = gmm.gmm_params_pattern.fold(
            gmm_params_free + pert, free=True)

        log_lik_by_nk, e_z = rm_lib.wrap_get_loglik_terms(
            gmm_params, gmm.transformed_reg_params)
        log_prior = rm_lib.get_log_prior(
            gmm_params['centroids'], gmm_params['probs'], gmm.prior_params)

        assert_array_almost_equal(e_z, rm_lib.get_e_z(log_lik_by_nk))
        assert_array_almost_equal(np.ones(e_z.shape[0]), np.sum(e_z, axis=1))

        # Neither log_lik_by_nk nor log_prior depend on e_z.
        def kl_ez(e_z):
            return rm_lib.get_kl(log_lik_by_nk, e_z, log_prior)
        e_z_pattern = paragami.SimplexArrayPattern(
            array_shape=(e_z.shape[0], ), simplex_size=e_z.shape[1])
        kl_ez_flat = paragami.FlattenFunctionInput(
            kl_ez, patterns=e_z_pattern, free=True)

        get_e_z_grad = autograd.grad(kl_ez_flat)
        e_z_grad = get_e_z_grad(e_z_pattern.flatten(e_z, free=True))
        assert_array_almost_equal(np.zeros_like(e_z_grad), e_z_grad)

    def test_sparse_derivs(self):
        regs = generate_test_fit.load_test_regs()
        comb_params, metadata = generate_test_fit.load_test_fit()
        gmm = generate_test_fit.load_test_gmm(regs, comb_params['reg'])

        opt_reg_params = comb_params['reg']
        opt_gmm_params = comb_params['mix']
        fit_derivs = sens_lib.FitDerivatives(
            opt_gmm_params, opt_reg_params,
            gmm.gmm_params_pattern, regs.reg_params_pattern,
            gmm=gmm, regs=regs)

        gmm_slice = np.ix_(fit_derivs.gmm_inds, fit_derivs.gmm_inds)
        reg_slice = np.ix_(fit_derivs.reg_inds, fit_derivs.reg_inds)
        cross_slice1 = np.ix_(fit_derivs.reg_inds, fit_derivs.gmm_inds)
        cross_slice2 = np.ix_(fit_derivs.gmm_inds, fit_derivs.reg_inds)

        comb_params_flat = fit_derivs.comb_params_pattern.flatten(
            { 'mix': opt_gmm_params, 'reg': opt_reg_params }, free=True)

        sp_hess = fit_derivs.full_hess.todense()

        # Test the GMM part.
        def get_gmm_objective(comb_params):
            return gmm.get_reg_params_kl(
                comb_params['mix'], comb_params['reg'])

        get_gmm_objective_flat = paragami.FlattenFunctionInput(
            get_gmm_objective,
            patterns=fit_derivs.comb_params_pattern, free=True)
        get_gmm_kl_hessian = autograd.hessian(get_gmm_objective_flat)

        full_hess = get_gmm_kl_hessian(comb_params_flat)

        assert_array_almost_equal(full_hess[gmm_slice], sp_hess[gmm_slice])
        assert_array_almost_equal(
            full_hess[cross_slice2], sp_hess[cross_slice2])

        assert_array_almost_equal(
            np.zeros((len(fit_derivs.reg_inds),
                      len(fit_derivs.gmm_inds))),
            sp_hess[cross_slice1])

        # Test the regression part.
        reg_hess = np.array(
            regs.get_sparse_free_hessian(
                fit_derivs.reg_params_free).todense())
        assert_array_almost_equal(reg_hess, sp_hess[reg_slice])

        # Test to and from JSON.
        fit_derivs_json = fit_derivs.to_json()
        json_fit_derivs = \
            sens_lib.FitDerivatives.from_json(fit_derivs_json)

        utils_for_testing_lib.assert_fit_derivs_equal(
            self, fit_derivs, json_fit_derivs)

        # Test to and from a numpy dict
        fit_derivs_dict = fit_derivs.to_numpy_dict()
        dict_fit_derivs = sens_lib.FitDerivatives.from_numpy_dict(
            fit_derivs_dict)
        utils_for_testing_lib.assert_fit_derivs_equal(
            self, fit_derivs, dict_fit_derivs)

    def test_json(self):
        regs = generate_test_fit.load_test_regs()
        comb_params, metadata = generate_test_fit.load_test_fit()
        gmm = generate_test_fit.load_test_gmm(regs, comb_params['reg'])

        preconditioner = 3 * np.eye(
            gmm.gmm_params_pattern.flat_length(free=True))
        gmm.get_kl_conditioned.set_preconditioner_matrix(preconditioner)
        gmm_json = gmm.to_json()
        new_gmm = rm_lib.GMM.from_json(gmm_json, regs, comb_params['reg'])

        utils_for_testing_lib.assert_gmm_equal(self, gmm, new_gmm)

        gmm_params_pattern = rm_lib.get_gmm_params_pattern(
            gmm.obs_dim, gmm.num_components)
        gmm_params = gmm_params_pattern.random()

        timepoints = np.array([1, 2, 3])
        metadata = saving_gmm_utils.get_result_metadata(
            timepoints, description='hello')


if __name__ == '__main__':
    unittest.main()
