#!/usr/bin/env python3

# Generate and save data for testing.  Run this from the tests directory
# before running the unit tests.

import numpy as np
import os

from aistats2019_ij_paper import regression_lib as reg_lib
from aistats2019_ij_paper import regression_mixture_lib as rm_lib
from aistats2019_ij_paper import sensitivity_lib as sens_lib
from aistats2019_ij_paper import saving_gmm_utils as save_lib
from aistats2019_ij_paper import spline_bases_lib

from aistats2019_ij_paper.tests import utils_for_testing_lib

import aistats2019_ij_paper
module_path = os.path.dirname(aistats2019_ij_paper.__file__)
test_data_dir = os.path.join(module_path, 'tests/data')
test_regression_file = os.path.join(test_data_dir, 'regs.json')
test_gmm_file = os.path.join(test_data_dir, 'gmm.json')
test_opt_file = os.path.join(test_data_dir, 'opt.npz')
test_full_opt_file = os.path.join(test_data_dir, 'full_opt.npz')

def run_script_message():
    raise ValueError(
        'You must generate test files with generate_test_fit.py first.')

def load_test_gmm(regs, reg_params):
    if not os.path.isfile(test_gmm_file):
        run_script_message()
    with open(test_gmm_file) as infile:
        gmm_json = infile.read()
    return rm_lib.GMM.from_json(gmm_json, regs, reg_params)

def load_test_regs():
    if not os.path.isfile(test_regression_file):
        run_script_message()
    with open(test_regression_file) as infile:
        regs_json = infile.read()
    return reg_lib.Regressions.from_json(regs_json)

def load_test_fit():
    if not os.path.isfile(test_opt_file):
        run_script_message()
    comb_params_free, comb_params_pattern, metadata = \
        save_lib.load_refit(test_opt_file)
    return comb_params_pattern.fold(
        comb_params_free, free=True), metadata

def load_test_derivs():
    if not os.path.isfile(test_full_opt_file):
        run_script_message()
    return save_lib.load_initial_optimum(test_full_opt_file)

if __name__ == '__main__':
    np.random.seed(42)

    # Generate data.
    true_k = 4
    n_obs = 12
    timepoints = utils_for_testing_lib.get_sample_timepoints(reps=3)

    true_df = 4
    regressors = spline_bases_lib.get_genomics_spline_basis(
        timepoints, df=true_df, degree=3)

    r = regressors.shape[1]
    # Set the y_scale large to make uncertainty in the z.
    y, true_z_ind, true_z, true_beta, true_a = \
        utils_for_testing_lib.simulate_data(
            x=regressors,
            y_scale=5,
            true_k=true_k,
            n_obs=n_obs,
            beta_prior_mean=np.zeros(r),
            beta_prior_cov=np.eye(r) * 10,
            log_shift_mean=3,
            log_shift_scale=2)

    # Define and fit a regression object.
    regs = reg_lib.Regressions(y, regressors)
    opt_reg_params = regs.get_optimal_regression_params()
    with open(test_regression_file, 'w') as outfile:
        outfile.write(regs.to_json())

    # Define a gmm object.
    num_components = 5

    trans_obs_dim = regs.x.shape[1] - 1
    prior_params = \
        rm_lib.get_base_prior_params(trans_obs_dim, num_components)

    prior_params['probs_alpha'][:] = 1
    prior_params['centroid_prior_info'] = 1e-5 * np.eye(trans_obs_dim)
    gmm = rm_lib.GMM(num_components, prior_params, regs, opt_reg_params,
                     inflate_coef_cov=None,
                     cov_regularization=0.1)
    with open(test_gmm_file, 'w') as outfile:
        outfile.write(gmm.to_json())

    # Fit and save.
    init_params = \
        rm_lib.kmeans_init(gmm.transformed_reg_params,
                           gmm.num_components, 50)
    init_params_flat = gmm.gmm_params_pattern.flatten(init_params, free=True)

    gmm_opt, gmm_opt_x = gmm.optimize_fully(
        init_params_flat, verbose=True, gtol=1e-8)
    opt_gmm_params = gmm.gmm_params_pattern.fold(gmm_opt_x, free=True)

    comb_params_pattern = sens_lib.get_combined_parameters_pattern(
        reg_params_pattern=regs.reg_params_pattern,
        gmm_params_pattern=gmm.gmm_params_pattern)
    comb_params = { 'mix': opt_gmm_params, 'reg': opt_reg_params }
    comb_params_free = comb_params_pattern.flatten(comb_params, free=True)

    save_lib.save_refit(
        outfile=test_opt_file,
        comb_params_free=comb_params_free,
        comb_params_pattern=comb_params_pattern,
        initial_fit_infile='test',
        time_w=regs.time_w,
        lo_inds=[],
        full_lo_inds=[],
        extra_metadata={'timepoints': timepoints})

    # Get the sensitivity matrices.
    fit_derivs = sens_lib.FitDerivatives(
        opt_gmm_params, opt_reg_params,
        gmm.gmm_params_pattern, regs.reg_params_pattern,
        gmm=gmm, regs=regs,
        verbose=False, print_every=0)

    extra_metadata = {'note': 'for testing'}
    save_lib.save_initial_optimum(
        test_full_opt_file,
        gmm=gmm,
        regs=regs,
        fit_derivs=fit_derivs,
        timepoints=timepoints,
        extra_metadata=extra_metadata)

    regs = load_test_regs()
    comb_params, metadata = load_test_fit()
    load_test_gmm(regs, comb_params['reg'])
    load_test_derivs()
