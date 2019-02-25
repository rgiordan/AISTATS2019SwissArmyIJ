#!/usr/bin/env python3

"""Run and save the initial fit along with Hessian information.

Example:
./initial_fit.py --df 4 --num_components 18
"""

import argparse
import numpy as np
import os
import sys
import time

from aistats2019_ij_paper import regression_mixture_lib as rm_lib
from aistats2019_ij_paper import regression_lib as reg_lib
from aistats2019_ij_paper import sensitivity_lib as sens_lib
from aistats2019_ij_paper import spline_bases_lib
from aistats2019_ij_paper import loading_data_utils
from aistats2019_ij_paper import saving_gmm_utils


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)

# results folders
parser.add_argument('--outfolder', default='../fits', type=str)
parser.add_argument('--test_out_filename', default=None, type=str)
parser.add_argument('--out_filename', default=None, type=str)
parser.add_argument('--train_indx_file',
                    default='../fits/train_indx.npy', type=str)

# Set genomic_time_series_dir should be the location of a clone of the repo
# https://github.com/NelleV/genomic_time_series_bnp
parser.add_argument('--genomic_time_series_dir',
                    default='../../genomic_time_series_bnp/', type=str)

# model parameters
parser.add_argument('--df', type=int, required=True)
parser.add_argument('--degree', default=3, type=int)
parser.add_argument('--num_components', default=18, type=int)

parser.add_argument('--force', default='False', type=str,
                    help='Set to ``True`` to overwrite results.')

args = parser.parse_args()

np.random.seed(args.seed)

assert os.path.exists(args.outfolder)
assert os.path.exists(args.genomic_time_series_dir)

if args.out_filename is None:
    args.out_filename = saving_gmm_utils.get_initial_fit_filename(
        args.df, args.degree, args.num_components)

if args.test_out_filename is None:
    args.test_out_filename = \
        saving_gmm_utils.get_test_regs_filename(args.df, args.degree)

full_out_filename = os.path.join(args.outfolder, args.out_filename)
test_out_full_filename = os.path.join(args.outfolder, args.test_out_filename)

if os.path.isfile(full_out_filename) and \
        os.path.isfile(test_out_full_filename):

    err_msg = 'Destinations {} and {} already exist.  '.format(
        full_out_filename, test_out_full_filename)
    if args.force == 'True':
        print(err_msg + '  Overwriting.')
    else:
        raise ValueError(err_msg + 'To overwrite, set the flag --force True')


print('Results will be saved to {}'.format(full_out_filename))

#############
# load data #
#############

y_train, y_test, train_indx, timepoints = \
    loading_data_utils.load_genomics_data(
        args.genomic_time_series_dir,
        split_test_train = True,
        train_indx_file = args.train_indx_file)

# Record which indices of the original data were train and which were test.
n_train = np.shape(y_train)[0]
n_test = np.shape(y_test)[0]

n_genes = n_train + n_test

test_indx = np.setdiff1d(np.arange(n_genes), train_indx)
gene_indx = np.concatenate((train_indx, test_indx))

###############
# Regressions #
###############

regressors = spline_bases_lib.get_genomics_spline_basis(
    timepoints, df=args.df, degree=args.degree)

# Define and save data for the test regressions.  The training
# regressions will be saved below with the rest of the fit.
regs_test = reg_lib.Regressions(y_test, regressors)
with open(test_out_full_filename, 'w') as outfile:
    outfile.write(regs_test.to_json())

# Run the regression.
print('Running regressions...')
regs = reg_lib.Regressions(y_train, regressors)
reg_time = time.time()
opt_reg_params = regs.get_optimal_regression_params()
reg_time = time.time() - reg_time

#################
# Mixture model #
#################

# Define prior parameters.
num_components = args.num_components
trans_obs_dim = regs.x.shape[1] - 1
prior_params = \
    rm_lib.get_base_prior_params(trans_obs_dim, num_components)
prior_params['probs_alpha'][:] = 1
prior_params['centroid_prior_info'] = 1e-5 * np.eye(trans_obs_dim)

gmm = rm_lib.GMM(args.num_components,
                 prior_params, regs, opt_reg_params,
                 inflate_coef_cov=None,
                 cov_regularization=0.1)

print('Running k-means init...')
init_gmm_params = \
    rm_lib.kmeans_init(gmm.transformed_reg_params,
                       gmm.num_components, 50)
init_x = gmm.gmm_params_pattern.flatten(init_gmm_params, free=True)


print('Fitting GMM...')
print('\tRunning gmm initial optimum...')
gmm.conditioned_obj.set_print_every(1)

opt_time = time.time()
gmm_opt, init_x2 = gmm.optimize(init_x, gtol=1e-2)

print('\tUpdating preconditioner...')
kl_hess = gmm.update_preconditioner(init_x2)

print('\tRunning preconditioned optimization...')
gmm.conditioned_obj.reset()
gmm_opt, gmm_opt_x = gmm.optimize_fully(init_x2, verbose=True)
print(gmm_opt['gmm_opt_cond'].message)
opt_time = time.time() - opt_time

opt_gmm_params = gmm.gmm_params_pattern.fold(gmm_opt_x, free=True)
print('Fit is done.')

print('Getting fit derivatives...')
hess_time = time.time()
fit_derivs = sens_lib.FitDerivatives(
    opt_gmm_params, opt_reg_params,
    gmm.gmm_params_pattern, regs.reg_params_pattern,
    gmm=gmm, regs=regs,
    print_every=10)
hess_time = time.time() - hess_time
print('Done.')

print('Saving...')

extra_metadata = dict()
extra_metadata['opt_time'] = opt_time
extra_metadata['reg_time'] = reg_time
extra_metadata['hess_time'] = hess_time
extra_metadata['df'] = args.df
extra_metadata['degree'] = args.degree

saving_gmm_utils.save_initial_optimum(
    full_out_filename,
    gmm=gmm,
    regs=regs,
    timepoints=timepoints,
    fit_derivs=fit_derivs,
    extra_metadata=extra_metadata)

print('All done!')
