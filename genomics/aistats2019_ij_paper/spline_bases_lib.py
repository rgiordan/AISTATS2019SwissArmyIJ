import numpy as np
import scipy as sp
import patsy


def get_bspline_design_matrix(timepoints, n_knots, df, degree):
    """
    Use ``patsy`` to get a spline basis for a given set of timepoints.

    Parameters
    -------------
    timepoints :
        The times of the observed gene expressions.
    n_knots : `int`
        specifies the number of knots, which will be equally spaced
        between min(timepoints) and max(timepoints)
    df : `int`
        The degrees of freedom of the spline basis.
    degree : `int`
        The degree of the spline basis.
    """

    knots = np.linspace(np.min(timepoints), np.max(timepoints), n_knots)
    knots = {"x": knots}
    timepoints_dict = {"x": timepoints}
    design_string = \
        "bs(x, df={}, degree={}, include_intercept=True) - 1".format(df, degree)
    design_matrix = patsy.dmatrix(design_string, knots)
    x_bs = patsy.build_design_matrices(
        [design_matrix.design_info], timepoints_dict)[0]
    return x_bs


def get_genomics_spline_basis(timepoints, exclude_num=3, df=7, degree=3):
    """
    A function to get a special regression basis for the genomics time series
    data with a smooth spline in the early times and distinct levels for the
    later times.
    """
    unique_timepoints = np.unique(timepoints)
    spline_timepoints = unique_timepoints[:-exclude_num]

    regressors_spline = get_bspline_design_matrix(
            timepoints=spline_timepoints, n_knots=df, df=df, degree=degree)

    # Give the excluded times each their own level.
    regressors_unique = sp.linalg.block_diag(
        regressors_spline, np.eye(exclude_num))

    # There are repeated timepoints, so repeat the rows.
    timepoint_matching = [ \
        np.argwhere(unique_timepoints == t)[0,0] for t in timepoints ]
    regressors = regressors_unique[timepoint_matching, :]

    ev = np.linalg.eigvals(np.matmul(regressors.T, regressors))
    if np.sum(ev > 1e-6) < len(ev):
        warnings.warn('Only {} eigenvaluse were nonzero (of {})'.format(
            np.sum(ev > 1e-6), len(ev)))

    return regressors
