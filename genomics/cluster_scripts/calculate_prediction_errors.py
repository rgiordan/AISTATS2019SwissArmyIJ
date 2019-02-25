#!/usr/bin/env python3
"""Calculate and save prediction errors from all the refit data.
This uses the output of ``intial_fit.py`` and ``refit.py``.

Example usage:
./calculate_prediction_errors.py \
    --df 4 \
    --lo_num_times 1 \
    --init_method warm
"""

import itertools
import numpy as np
import os
import paragami
import vittles
import scipy as sp
from scipy import sparse
import time

np.random.seed(3452453)

from aistats2019_ij_paper import regression_lib as reg_lib
from aistats2019_ij_paper import sensitivity_lib as sens_lib
from aistats2019_ij_paper import saving_gmm_utils
from aistats2019_ij_paper import mse_utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--outfolder', default='../fits', type=str)
parser.add_argument('--outfile', default=None, type=str)

parser.add_argument('--df', type=int, required=True)
parser.add_argument('--degree', default=3, type=int)
parser.add_argument('--num_components', default=18, type=int)

parser.add_argument('--lo_num_times', type=int, required=True)
parser.add_argument('--init_method', type=str, required=True)
parser.add_argument('--lo_max_num_timepoints', default=7, type=int)


class ErrorArrays:
    """A class for gathering and saving prediction errors in a format
    easily converted to a tidy dataframe.
    """
    def __init__(self, name):
        self._name = name
        self.error_mat = []
        self.orig_error_mat = []
        self.full_lo_inds_mat = []
        self.test_arr = []
        self.comb_arr = []
        self.row_arr = []
        self.rereg_arr = []
        self.method_arr = []

    def append_result(self, error, orig_error,
                      test, comb, rereg, full_lo_inds, method):
        self.error_mat.append(error)
        nrow = error.shape[0]
        self.orig_error_mat.append(orig_error)
        assert orig_error.shape[0] == nrow
        self.full_lo_inds_mat.append(
            np.repeat(np.expand_dims(full_lo_inds, 0), nrow, axis=0))
        self.test_arr.append([test] * nrow)
        self.comb_arr.append([comb] * nrow)
        self.rereg_arr.append([rereg] * nrow)
        self.method_arr.append([method] * nrow)
        self.row_arr.append(np.arange(nrow))

    def get_result_dict(self):
        res = dict()
        res[self._name + '_error_mat'] = \
            np.vstack(self.error_mat)
        res[self._name + '_orig_error_mat'] = \
            np.vstack(self.orig_error_mat)
        res[self._name + '_full_lo_inds_mat'] = \
            np.vstack(self.full_lo_inds_mat)
        # Python's 1d matrices are row vectors by default, so hstack.
        res[self._name + '_test_arr'] = np.hstack(self.test_arr)
        res[self._name + '_comb_arr'] = np.hstack(self.comb_arr)
        res[self._name + '_rereg_arr'] = np.hstack(self.rereg_arr)
        res[self._name + '_method_arr'] = np.hstack(self.method_arr)
        res[self._name + '_row_arr'] = np.hstack(self.row_arr)
        return res




###############
# Script

if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.isdir(args.outfolder):
        raise ValueError('{} does not exist.'.format(args.outfolder))

    if args.outfile is None:
        args.outfile = saving_gmm_utils.get_prediction_error_filename(
            df=args.df,
            degree=args.degree,
            num_components=args.num_components,
            lo_num_times=args.lo_num_times,
            init_method=args.init_method,
            lo_max_num_timepoints=args.lo_max_num_timepoints)

    def get_refit_filename(lo_which_comb):
        """Get a refit filename using the specified arguments.
        """
        return saving_gmm_utils.get_refit_filename(
            df=args.df, degree=args.degree,
            num_components=gmm.num_components,
            lo_num_times=args.lo_num_times,
            lo_which_comb=lo_which_comb,
            lo_max_num_timepoints=args.lo_max_num_timepoints,
            init_method=args.init_method)

    ###############################
    # Load the original fit.

    print('Loading original fit.')
    npz_infile = \
        saving_gmm_utils.get_initial_fit_filename(
            df=args.df,
            num_components=args.num_components,
            degree=args.degree)
    full_fit, gmm, regs, initial_metadata = \
        saving_gmm_utils.load_initial_optimum(
            os.path.join(args.outfolder, npz_infile))

    opt_comb_params = full_fit.get_comb_params()

    ###############################
    # Load the test data
    test_regression_infile = saving_gmm_utils.get_test_regs_filename(
        df=args.df, degree=args.degree)
    with open(os.path.join(args.outfolder, test_regression_infile), 'r') as infile:
        regs_test = reg_lib.Regressions.from_json(infile.read())
    reg_params_test = regs_test.get_optimal_regression_params()

    ###############################
    # Make a sensitivity object.

    # If you don't cast the jacobian to an array from
    # a matrix, the output is a 2d-array.
    weight_sens = vittles.HyperparameterSensitivityLinearApproximation(
        objective_fun=lambda: 0,
        opt_par_value=full_fit.comb_params_free,
        hyper_par_value=regs.time_w,
        hessian_at_opt=sp.sparse.csc_matrix(full_fit.full_hess),
        cross_hess_at_opt=np.array(full_fit.t_jac.todense()))

    ###############################
    # Check that all the files are there.

    num_comb = int(sp.special.comb(
        args.lo_max_num_timepoints, args.lo_num_times))

    missing_files = 0
    for lo_which_comb in range(num_comb):
        load_filename = get_refit_filename(lo_which_comb)
        if not os.path.isfile(os.path.join(
                args.outfolder, load_filename)):
            print('{} is missing'.format(load_filename))
            missing_files += 1

    if missing_files > 0:
        raise ValueError('Missing {} of {} files.'.format(
            missing_files, num_comb))

    ########################
    # Load the refits.

    print('Loading the refits.')

    time_w_arr = []
    comb_params_free_ref_arr = []
    comb_params_free_lin_arr = []
    full_lo_inds_arr = []

    lr_time = 0
    refit_time = 0
    load_time = time.time()
    for lo_which_comb in range(num_comb):
        print('{} of {}'.format(lo_which_comb, num_comb))
        load_filename = \
            saving_gmm_utils.get_refit_filename(
                df=args.df, degree=args.degree,
                num_components=gmm.num_components,
                lo_num_times=args.lo_num_times,
                lo_which_comb=lo_which_comb,
                lo_max_num_timepoints=args.lo_max_num_timepoints,
                init_method=args.init_method)

        comb_params_free_ref, comb_params_pattern_refit, refit_metadata = \
            saving_gmm_utils.load_refit(
                os.path.join(args.outfolder, load_filename))
        refit_time += refit_metadata['opt_time']

        assert(comb_params_pattern_refit == full_fit.comb_params_pattern)
        time_w = refit_metadata['time_w']
        lo_inds = refit_metadata['lo_inds']
        full_lo_inds = refit_metadata['full_lo_inds']

        time_w_arr.append(time_w)
        full_lo_inds_arr.append(full_lo_inds)
        comb_params_free_ref_arr.append(comb_params_free_ref)

        # Get the linear repsonse prediction.
        tic = time.time()
        comb_params_free_lin = \
            weight_sens.predict_opt_par_from_hyper_par(time_w)
        lr_time += time.time() - tic
        comb_params_free_lin_arr.append(comb_params_free_lin)
        load_time = time.time() - load_time


    ###############################
    # Calculate and save the test error.

    print('Calculating test error.')

    get_lo_err = paragami.FlattenFunctionInput(
        mse_utils.get_lo_err_folded,
        patterns=full_fit.comb_params_pattern, free=True)
    get_rereg_lo_err = paragami.FlattenFunctionInput(
        mse_utils.get_rereg_lo_err_folded,
        patterns=full_fit.comb_params_pattern, free=True)

    def get_lo_err_opts(comb_params_free, test, rerun_reg,
                        keep_inds, time_w=None):
        if (not rerun_reg) and (time_w is not None):
            raise ValueError(
                'Do not specify time_w without ``rerun_reg``')

        if test:
            this_regs = regs_test
            this_reg_params = reg_params_test
        else:
            this_regs = regs
            this_reg_params = opt_comb_params['reg']
        if rerun_reg:
            # Re-regress and cluster the new regressions.
            return get_rereg_lo_err(
                    comb_params_free,
                    keep_inds,
                    time_w,
                    this_regs,
                    gmm)
        else:
            # Use the given regression parameters.
            return get_lo_err(
                    comb_params_free,
                    keep_inds,
                    this_regs,
                    this_reg_params,
                    gmm)

    refit_lo_errs = ErrorArrays('refits')

    tf = [True, False]
    test_err_time = time.time()
    print('Getting test errors.')
    for test, rereg in itertools.product(tf, tf):
        # Get the original errors.
        if rereg:
            time_w = np.ones(regs.y.shape[1])
        else:
            time_w = None
        all_inds = np.arange(regs.y.shape[1])
        orig_error = get_lo_err_opts(
            full_fit.comb_params_free,
            test=test,
            rerun_reg=rereg,
            keep_inds=all_inds,
            time_w=time_w)

        for comb in range(num_comb):
            full_lo_inds = full_lo_inds_arr[comb]
            if rereg:
                time_w = time_w_arr[comb]
            else:
                time_w = None

            error = get_lo_err_opts(
                comb_params_free_ref_arr[comb],
                test=test,
                rerun_reg=rereg,
                keep_inds=full_lo_inds,
                time_w=time_w)

            refit_lo_errs.append_result(
                error=error,
                orig_error=orig_error[:, full_lo_inds],
                test=test,
                comb=comb,
                rereg=rereg,
                full_lo_inds=full_lo_inds,
                method='ref')

            error = get_lo_err_opts(
                comb_params_free_lin_arr[comb],
                test=test,
                rerun_reg=rereg,
                keep_inds=full_lo_inds,
                time_w=time_w)

            refit_lo_errs.append_result(
                error=error,
                orig_error=orig_error[:, full_lo_inds],
                test=test,
                comb=comb,
                rereg=rereg,
                full_lo_inds=full_lo_inds,
                method='lin')
        test_err_time = time.time() - test_err_time

    print('Saving.')
    save_dict = dict()
    save_dict.update(refit_lo_errs.get_result_dict())
    save_dict['num_comb'] = num_comb
    save_dict['lr_time'] = lr_time
    save_dict['refit_time'] = refit_time
    save_dict['gmm_param_length'] = \
        full_fit.comb_params_pattern['mix'].flat_length(free=TRUE)
    save_dict['reg_param_length'] = \
        full_fit.comb_params_pattern['reg'].flat_length(free=TRUE)
    save_dict.update(initial_metadata)

    np.savez_compressed(
        os.path.join(args.outfolder, args.outfile), **save_dict)
