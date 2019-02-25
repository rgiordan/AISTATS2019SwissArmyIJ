#!/usr/bin/env python3

"""Load a previous fit and refit, possibly with new time weights.
This uses the output of the script ``initial_fit.py``.

Example usage:
./refit.py \
    --df 4 \
    --lo_num_times 1 \
    --lo_which_comb 0 \
    --init_method warm
"""

import argparse
from copy import deepcopy
import numpy as np
import os
import sys
import time

from aistats2019_ij_paper import regression_mixture_lib as rm_lib
from aistats2019_ij_paper import saving_gmm_utils
from aistats2019_ij_paper import mse_utils

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)

# results folders
parser.add_argument('--outfolder', default='../fits', type=str)
parser.add_argument('--out_filename', default=None, type=str)

# Specify either the initinal fit file or the df, degree, and num_components.
parser.add_argument('--initial_fit_infile', default=None, type=str)

parser.add_argument('--df', default=None, type=int)
parser.add_argument('--num_components', default=18, type=int)
parser.add_argument('--degree', default=3, type=int)


parser.add_argument('--lo_num_times', type=int, required=True)
parser.add_argument('--lo_which_comb', type=int)
parser.add_argument('--lo_max_num_timepoints', default=7, type=int)

parser.add_argument('--init_method', type=str, required=True)

parser.add_argument('--force', default='False', type=str,
                    help='Set to ``True`` to overwrite results.')


args = parser.parse_args()

np.random.seed(args.seed)

if not os.path.exists(args.outfolder):
    raise ValueError('outfolder {} does not exist'.format(args.outfolder))

if args.initial_fit_infile is None:
    if (args.df is None) or \
        (args.num_components is None) or \
        (args.degree is  None):
        raise ValueError(
            'If ``initial_fit_infile`` is not specified, ' +
            'you must specify ``df``, ``degree``, and ``num_components``.')
    args.initial_fit_infile = os.path.join(
        args.outfolder,
        saving_gmm_utils.get_initial_fit_filename(
            args.df, args.degree, args.num_components))

if not os.path.isfile(args.initial_fit_infile):
    raise ValueError('initial_fit_infile {} does not exist'.format(
        args.initial_fit_infile))

valid_init_methods = ['kmeans', 'warm']
if not (args.init_method in valid_init_methods):
    raise ValueError(
        '``init_method`` must be one of {}'.format(valid_init_methods))

if args.out_filename is None:
    args.out_filename = saving_gmm_utils.get_refit_filename(
        df=args.df, degree=args.degree, num_components=args.num_components,
        lo_num_times=args.lo_num_times,
        lo_which_comb=args.lo_which_comb,
        lo_max_num_timepoints=args.lo_max_num_timepoints,
        init_method=args.init_method)
full_out_filename = os.path.join(args.outfolder, args.out_filename)
print('Results will be saved to {}'.format(full_out_filename))


if os.path.isfile(full_out_filename):
    err_msg = 'Destination {} already exists.  '.format(full_out_filename)
    if args.force == 'True':
        print(err_msg + '  Overwriting.')
    else:
        raise ValueError(err_msg + 'To overwrite, set the flag --force True')


#####################
# load previous fit #
#####################

print('Loading previous fit...')
comb_params_pattern, comb_params, gmm, regs, metadata = \
    saving_gmm_utils.load_initial_optimum_fit_only(args.initial_fit_infile)
gmm_params_free = \
    comb_params_pattern['mix'].flatten(
        comb_params['mix'], free=True)

timepoints = metadata['timepoints']
df = metadata['df']
degree = metadata['degree']
num_components = gmm.num_components


# Get the left-out indices and weights.
print('Getting left-out indices...')
lo_inds = mse_utils.get_indexed_combination(
    num_times=args.lo_num_times,
    which_comb=args.lo_which_comb,
    max_num_timepoints=args.lo_max_num_timepoints)
new_time_w, full_lo_inds = mse_utils.get_time_weight(lo_inds, timepoints)
print('Leaving out {}.'.format(lo_inds))

#########
# Refit #
#########

print('Re-optimizing.')
gmm.conditioned_obj.set_print_every(1)

if args.init_method == 'warm':
    print('Using warm start.')
    opt_time = time.time()
    reopt, gmm_params_free_w, reg_params_w = \
        rm_lib.refit_with_time_weights(
            gmm, regs, new_time_w,
            gmm_params_free)
    print(reopt['gmm_opt_cond'].message)
    opt_time = time.time() - opt_time

elif args.init_method == 'kmeans':
    print('Using k-means init.')
    regs.time_w = deepcopy(new_time_w)
    reg_params_w = regs.get_optimal_regression_params()
    gmm.set_regression_params(reg_params_w)

    init_gmm_params = \
        rm_lib.kmeans_init(gmm.transformed_reg_params,
                           gmm.num_components, 50)
    init_x = gmm.gmm_params_pattern.flatten(init_gmm_params, free=True)

    opt_time = time.time()
    gmm_opt, init_x2 = gmm.optimize(init_x, gtol=1e-2)

    print('\tUpdating preconditioner...')
    kl_hess = gmm.update_preconditioner(init_x2)

    print('\tRunning preconditioned optimization...')
    gmm.conditioned_obj.reset()
    reopt, gmm_params_free_w = gmm.optimize_fully(init_x2, verbose=True)
    print(gmm_opt.message)
    opt_time = time.time() - opt_time

else:
    raise ValueError('Unknown ``init_method`` {}'.format(args.init_method))

gmm_params_w = \
    comb_params_pattern['mix'].fold(gmm_params_free_w, free=True)

refit_comb_params = {
    'mix': gmm_params_w,
    'reg': reg_params_w }

refit_comb_params_free = \
    comb_params_pattern.flatten(refit_comb_params, free=True)

extra_metadata = dict()
extra_metadata['opt_time'] = opt_time
extra_metadata['df'] = df
extra_metadata['degree'] = degree
extra_metadata['num_components'] = num_components

print('Saving...')
saving_gmm_utils.save_refit(
    outfile=full_out_filename,
    comb_params_free=refit_comb_params_free,
    comb_params_pattern=comb_params_pattern,
    initial_fit_infile=args.initial_fit_infile,
    time_w=new_time_w,
    lo_inds=lo_inds,
    full_lo_inds=full_lo_inds,
    extra_metadata=extra_metadata)

print('All done!')
