#!/usr/bin/env python3

import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

import paragami

from aistats2019_ij_paper.tests import utils_for_testing_lib

import aistats2019_ij_paper.regression_lib as reg_lib
from aistats2019_ij_paper import spline_bases_lib


class TestRegression(unittest.TestCase):
    def test_regressions(self):
        np.random.seed(42)

        true_df = 4
        true_k = 4

        n_obs = 5

        timepoints = utils_for_testing_lib.get_sample_timepoints(reps=3)
        regressors = spline_bases_lib.get_genomics_spline_basis(
        timepoints, df=true_df, degree=3)

        r = regressors.shape[1]
        y, true_z_ind, true_z, true_beta, true_a = \
            utils_for_testing_lib.simulate_data(
                x=regressors,
                y_scale=1,
                true_k=true_k,
                n_obs=n_obs,
                beta_prior_mean=np.zeros(r),
                beta_prior_cov=np.eye(r) * 10,
                log_shift_mean=3,
                log_shift_scale=2)

        regs = reg_lib.Regressions(y, regressors)
        reg_params = regs.get_optimal_regression_params()

        # Check that set_regression_params sets optimal values for the KL
        # divergence.
        get_kl_objective = \
            paragami.OptimizationObjective(
                lambda x: regs.get_flat_kl(x))

        true_x = \
            regs.reg_params_pattern.flatten(reg_params, free=True)

        self.assertTrue(np.linalg.norm(get_kl_objective.grad(true_x)) <= 1e-8)

        # Check the sparse Hessian.
        print('Calculating sparse Hessian.')
        sp_hess = regs.get_sparse_free_hessian(true_x, print_every=0)
        print('Calculating dense Hessian.')
        full_hess = get_kl_objective.hessian(true_x)
        print('Done with Hessians.')
        assert_array_almost_equal(full_hess, np.array(sp_hess.todense()))

        # Check that the Jacobian can be calculated and has the right shape.
        jac = regs.get_weight_jacobian(true_x)
        self.assertEqual((len(timepoints), len(true_x)), jac.shape)

        regs = reg_lib.Regressions(y, regressors, reverse_jac_order=True)
        jac = regs.get_weight_jacobian(true_x)
        self.assertEqual((len(timepoints), len(true_x)), jac.shape)

        # Test converting to and from JSON.
        reg_json = regs.to_json()
        new_regs = reg_lib.Regressions.from_json(reg_json)

        utils_for_testing_lib.assert_regressions_equal(self, regs, new_regs)


if __name__ == '__main__':
    unittest.main()
