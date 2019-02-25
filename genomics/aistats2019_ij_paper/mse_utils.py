"""Utilities for calcualting and saving prediction error for GMM models.
"""

import numpy as np
import scipy as sp

import json_tricks
import json

import itertools

from copy import deepcopy

from . import regression_mixture_lib as rm_lib


def timepoint_to_int(orig_timepoints):
    """Convert the timepoints to integers with a warning if they were not
    round numbers to start with.
    """
    timepoints = []
    for t in orig_timepoints:
        t_int = int(t)
        if t_int != t:
            warnings.warn(
                'Non-integer timepoint {} being converted to integer.'.format(
                    t))
        timepoints.append(t_int)
    return timepoints


def get_time_weight(lo_inds, timepoints):
    """Get a weight vector leaving out a combination of unique timepoints.

    Parameters
    -----------
    lo_inds : `list` of `int`
        Indices into the /unique/ timepoints to leave out.
    timepoints : `list` of `int`
        Possibly non-unique timepoints.

    Returns
    --------
    time_w : `np.ndarray`
        A numeric array with zeros in every place where ``timepoints``
        matches ``np.unique(timepoints)[lo_inds]``.
    """

    timepoints_int = timepoint_to_int(timepoints)
    unique_timepoints = np.unique(timepoints_int)
    time_w = np.ones_like(timepoints)
    full_lo_inds = []
    for ind in lo_inds:
        matching_inds = (timepoints_int == unique_timepoints[ind])
        full_lo_inds.extend(list(np.argwhere(matching_inds).flatten()))
        time_w[matching_inds] = 0
    return time_w, np.array(full_lo_inds)


def get_indexed_combination(num_times, which_comb, max_num_timepoints):
    """Return one of the combinations of num_times items.

    This returns the ``which_comb``-th element of the set of all combinations
    of ``num_times`` elements taken from a set of length
    ``max_num_timepoints``.  The order of the combinations is determined
    by ``itertools.combinations`` and is always the same.

    Parameters
    ----------
    num_times: `int`
        The number of timepoints to leave out.  If 0, no timepoints are
        left out.
    which_comb: `int`
        The index into the combinations of ``num_times` left out points.
    max_num_timepoints: `int`
        The number of distinct timepoints it is possible to leave out.

    Returns
    --------
    leave_out_timepoint_inds: `numpy.ndarray` (N, )
        The zero-based indices of the elements left out in the corresponding
        combination.
    """

    if num_times < 0:
        raise ValueError('`num_times` must be a non-negative integer.')
    if num_times >= max_num_timepoints:
        raise ValueError(
            '`num_times` must be strictly less than `max_num_timepoints`.')
    num_comb = sp.special.comb(max_num_timepoints, num_times)
    if which_comb >= num_comb:
        raise ValueError(
             ('There are are {} combinations of {} selections from ' +
              '{} points.  The zero-based index `which_comb` must be ' +
              'less than this number.'.format(
                num_comb, num_times, max_num_timepoints)))

    if num_times == 0:
        # Leave out no timepoints.
        return np.array([])

    # Note that, because `range` is sorted, `itertools.combinations` is
    # also sorted, so this will always return the same order.
    combinations_iterator = itertools.combinations(
        range(max_num_timepoints), num_times)

    # This gets the `which_comb`-th element of the iterator as an array.
    leave_out_timepoint_inds = \
        np.array(next(
            itertools.islice(combinations_iterator, which_comb, None)))

    return leave_out_timepoint_inds


def get_predictions(gmm, gmm_params, reg_params):
    """Return a matrix of predictions the same shape as the observations.

    The predicted centroid is the expectation
    :math:`\sum_k \mathbb{E}[z_{nk}] \\beta_k`, not the most likely
    centroid.

    Parameters
    ----------
    gmm
    gmm_params
    transformed_reg_params
    """

    transformed_reg_params = \
        gmm.transform_regression_parameters(reg_params)
    e_z = rm_lib.wrap_get_e_z(gmm_params, transformed_reg_params)
    untransformed_preds = e_z @ gmm_params['centroids']
    return untransformed_preds @ gmm.unrotate_transform_mat.T


def get_prediction_error(gmm, gmm_params, regs, reg_params):
    """This gets the prediction error on the data contained in ``regs``,
    using a clustering of the regressions contained in ``reg_params``.
    """
    y = regs.y
    return \
        (y - np.mean(y, axis=1, keepdims=True)) - \
        get_predictions(gmm, gmm_params, reg_params)


def get_time_w_prediction_error(gmm, gmm_params, regs, time_w):
    """This re-runs the regressions using the weights time_w and
    returns the prediction errors on the new regressions.
    """
    if len(time_w) != len(regs.time_w):
        raise ValueError('``time_w`` is the wrong length.')
    regs.time_w = time_w
    new_reg_params = regs.get_optimal_regression_params()
    return get_prediction_error(
        gmm, gmm_params, regs, new_reg_params)


def get_lo_err_folded(comb_params, keep_inds, mse_regs, mse_reg_params, gmm):
    """Return the prediction error on the ``keep_inds``.  The
    data is taken from ``mse_regs``, and the parameters used for clustering
    are ``mes_reg_params``.
    """
    pred_err = get_prediction_error(
        gmm, comb_params['mix'], mse_regs, mse_reg_params)
    return pred_err[:, keep_inds]


def get_rereg_lo_err_folded(comb_params, keep_inds, time_w, mse_regs, gmm):
    """Return the prediction error on ``keep_inds``.  The
    data is taken from ``mse_regs``, and the parameters used for clustering
    are the exact regressions with weights ``time_w``.
    """
    mse_reg_params = mse_regs.get_optimal_regression_params(time_w)
    return get_lo_err_folded(
        comb_params, keep_inds, mse_regs, mse_reg_params, gmm)
