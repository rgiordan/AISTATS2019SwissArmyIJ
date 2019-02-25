import autograd
import autograd.numpy as np

import json

import paragami
import vittles

import scipy as osp


def run_regressions(y, regressors, w=None):
    """Get the optimal regression lines in closed form.

    Parameters
    ----------------
    y : `numpy.ndarray` (N, M)
        A matrix containing the outcomes of ``N`` regressions with
        ``M`` observations each.
    regressors : `numpy.ndarray` (M, D)
        A matrix of regressors.  The regression coefficient will be
        a ``D``-length vector.
    w : `numpy.ndarray` (M,), optional
        A vector of weights on the columns of ``y``.  If not set,
        the vector of ones is used.

    Returns
    ---------
    beta : `numpy.ndarray` (N, D)
        An array of the ``N`` regression coefficients.
    beta_infos : `numpy.ndarray` (N, D, D)
        An array of the "information" matrices, i.e. the inverse
        covariance matrices, of ``beta``.
    y_infos : `numpy.ndarray` (N,)
        An array of the inverse residual variances for each regression.
    """
    if w is None:
        w = np.ones(y.shape[1])
    assert y.shape[1] == regressors.shape[0]
    num_obs = y.shape[0]
    x_obs_dim = regressors.shape[1]
    y_obs_dim = y.shape[1]
    assert y_obs_dim > x_obs_dim

    beta = np.full((num_obs, x_obs_dim), float('nan'))
    beta_infos = np.full((num_obs, x_obs_dim, x_obs_dim), float('nan'))
    y_infos = np.full(num_obs, float('nan'))

    rtr = np.matmul(regressors.T, w[:, np.newaxis] * regressors)
    evs = np.linalg.eigvals(rtr)
    if np.min(evs < 1e-6):
        raise ValueError('Regressors are approximately singular.')
    rtr_inv_rt = np.linalg.solve(rtr, regressors.T)
    for n in range(num_obs):
        beta_reg = np.matmul(rtr_inv_rt, w * y[n, :])
        beta[n, :] = beta_reg
        resid = y[n, :] - np.matmul(regressors, beta_reg)

        # The extra -x_obs_dim comes from the beta variance -- see notes.
        #y_info = (y_obs_dim - x_obs_dim) / np.sum(resid ** 2)
        y_info = (sum(w) - x_obs_dim) / np.sum(w * (resid ** 2))

        beta_infos[n, :, :] = rtr * y_info
        y_infos[n] = y_info

    return beta, beta_infos, y_infos

#########################################################
# The objectives that ``run_regressions`` is optimizing.
#
# These are not actually used to calcualte the regressions because
# the regressions are available in closed form.  However, they are
# used to calcualte the sensitivity.

def get_regression_log_lik_by_nt(y, x, beta_mean, beta_info, y_info):
    resid = y - np.einsum('ni,ti->nt', beta_mean, x)
    beta_cov = np.linalg.inv(beta_info)
    log_lik_by_nt = \
        -0.5 * y_info[:, None] * (
            resid ** 2 + np.einsum('nij,tj,ti->nt', beta_cov, x, x)) + \
        0.5 * np.log(y_info[:, None])
    return log_lik_by_nt


def get_regression_kl(y, x, beta_mean, beta_info, y_info, obs_w, time_w):
    num_obs = y.shape[0]
    log_lik_by_nt = get_regression_log_lik_by_nt(
        y, x, beta_mean, beta_info, y_info)
    # slogdet on arrays is not currently supported in forward diff.
    # entropy = -0.5 * np.linalg.slogdet(beta_info)[1]
    entropy = -0.5 * np.array(
        [ np.linalg.slogdet(beta_info[n, :, :])[1] for n in range(num_obs) ])
    kl_by_obs = -1 * np.einsum('nt,t->n', log_lik_by_nt, time_w)
    kl = np.sum(obs_w * (kl_by_obs - entropy))
    return kl


###################################################3
# Paragami patterns for the regressions and data.

def get_regression_array_pattern(num_obs, x_obs_dim):
    """Get a paragami pattern for an array of approximate regression posteriors.

    Parameters
    -------------
    num_obs : `int`
        The number of distinct regressions.
    x_obs_dim : `int`
        The dimensionality of the regressors.

    Returns
    ----------
    reg_params_pattern : A paragami pattern.

    The field ``beta_mean`` is the ordinary regression
    coefficient, and ``beta_info`` is the estimated inverse covariance
    matrix.  The field ``y_info`` contains a point estimate for the
    inverse variance of each regressoin's residual.
    """
    reg_params_pattern = paragami.PatternDict()
    reg_params_pattern['beta_mean'] = \
        paragami.NumericArrayPattern(shape=(num_obs, x_obs_dim))
    reg_params_pattern['beta_info'] = \
        paragami.pattern_containers.PatternArray(
            array_shape=(num_obs, ),
            base_pattern=\
                paragami.PSDSymmetricMatrixPattern(size=x_obs_dim))
    reg_params_pattern['y_info'] = \
        paragami.NumericVectorPattern(length=num_obs, lb=0.0)
    return reg_params_pattern


def _get_data_pattern(x_shape, y_shape):
    data_pattern = paragami.PatternDict()
    data_pattern['y'] = paragami.NumericArrayPattern(y_shape)
    data_pattern['x'] = paragami.NumericArrayPattern(x_shape)
    return data_pattern


class Regressions(object):
    """A class containing the data for a set of regressions, each with
    a common set of regressors.
    """
    def __init__(self, y, x, reverse_jac_order=False):
        """
        Parameters
        -----------------
        y : `numpy.ndarray` (N, M)
            A matrix containing the outcomes of ``N`` regressions with
            ``M`` observations each.
        regressors : `numpy.ndarray` (M, D)
            A matrix of regressors.  The regression coefficient will be
            a ``D``-length vector.
        reverse_jac_order : `bool`, optional
            Whether to reverse the order in which the objective derivatives
            with respect to the parameters and weights are calculated.
        """
        if y.shape[1] != x.shape[0]:
            raise ValueError('y must have as many columns as x has rows')

        self.num_obs = y.shape[0]
        self.y_obs_dim = y.shape[1]
        self.x_obs_dim = x.shape[1]

        self.x = x
        self.y = y

        self.reg_params_pattern = get_regression_array_pattern(
            self.num_obs, self.x_obs_dim)

        self.get_flat_kl = paragami.FlattenFunctionInput(
            lambda reg_params: \
                self.get_regression_params_kl(
                    reg_params, self.obs_w, self.time_w),
            free=True, patterns=self.reg_params_pattern, argnums=0)

        # Attributes for the sparse Hessian.
        self.sparsity_pattern = self._get_sparsity_pattern()
        self._get_sparse_hess = \
            vittles.SparseBlockHessian(
                self.get_flat_kl, self.sparsity_pattern)

        # Attributes for the time weight sensitivity.
        self._reverse_jac_order = reverse_jac_order
        self.get_flat_time_kl = paragami.FlattenFunctionInput(
            lambda reg_params, time_w: \
                self.get_regression_params_kl(reg_params, self.obs_w, time_w),
            free=True, patterns=self.reg_params_pattern, argnums=0)

        if self._reverse_jac_order:
            self.get_flat_time_kl_grad = autograd.grad(
                self.get_flat_time_kl, argnum=0)
            self.get_flat_time_kl_jacobian = autograd.jacobian(
                self.get_flat_time_kl_grad, argnum=1)
        else:
            self.get_flat_time_kl_grad = autograd.grad(
                self.get_flat_time_kl, argnum=1)
            self.get_flat_time_kl_jacobian = autograd.jacobian(
                self.get_flat_time_kl_grad, argnum=0)

        # Weights
        self.initialize_weights()

    def initialize_weights(self):
        """Set the time and observation weights to vectors of ones.
        """
        # TODO: it would be less error-prone to save the weights with
        # the regression params rather than as regression attributes.
        self.time_w = np.ones(self.y_obs_dim)
        self.obs_w = np.ones(self.num_obs)

    def get_optimal_regression_params(self, time_w=None):
        """Run the regressions and return a set of regression parameters.

        Each row of ``y`` is regressed on ``x``.  An approximate posterior
        is calculated.

        Parameters
        -----------
        time_w : `numpy.ndarray` (N,), optional
            A vector of weights for the times, i.e. for the columns of ``y``.
            If not specified, the attribute ``self.time_w`` is used.

        Returns
        -----------
        reg_params :
            A parameter dictionary that can be used with the paragmi
            pattern given by ``get_regression_array_pattern()``.
        """
        if time_w is None:
            time_w = self.time_w
        reg_params = self.reg_params_pattern.empty(valid=False)
        reg_params['beta_mean'], \
        reg_params['beta_info'], \
        reg_params['y_info'] = \
            run_regressions(self.y, self.x, time_w)
        return reg_params

    def get_regression_params_kl(self, reg_params, obs_w, time_w):
        """This is the objective function that get_optimal_regression_params()
        optimizes in closed form.
        """
        return \
            get_regression_kl(
                self.y, self.x,
                reg_params['beta_mean'],
                reg_params['beta_info'],
                reg_params['y_info'],
                obs_w, time_w)

    def _get_sparsity_pattern(self):
        """Get the Hessian sparsity pattern for use in the sparse Hessian
        in the format that can be passed to ``paragami.SparseBlockHessian``.
        """
        sparse_inds = []
        regs_bool = self.reg_params_pattern.empty_bool(False)
        # For each regression, set its parameters to true and append
        # which indices in the flat free parameter they depend on.
        for n in range(self.num_obs):
            regs_bool['beta_mean'][n, :] = True
            regs_bool['beta_info'][n, :, :] = True
            regs_bool['y_info'][n] = True
            inds = self.reg_params_pattern.flat_indices(regs_bool, free=True)
            regs_bool['beta_mean'][n, :] = False
            regs_bool['beta_info'][n, :, :] = False
            regs_bool['y_info'][n] = False
            sparse_inds.append(inds)
        sparse_inds = np.array(sparse_inds)
        return sparse_inds

    def get_sparse_free_hessian(
        self, reg_params_free, obs_w=None, print_every=0):
        """Get a sparse representation of the Hessian matrix of the KL
        divergence at ``reg_params_free`` and ``obs_w``.

        Parameters
        --------------
        reg_params_free : `numpy.ndarray`
            A flat free value for the regression parameters.
        obs_w : `numpy.ndarray` (N,), optional
            A vector of observation weights.

        Returns
        -------------
        sp_hess : `scipy.sparse.csc_matrix`
            The Hessian of the regression's KL divergence objective function
            evaluated at ``reg_params_free``.
        """
        if obs_w is None:
            obs_w = self.obs_w
        sp_hess = self._get_sparse_hess.get_block_hessian(
            reg_params_free, print_every=print_every)
        return sp_hess

    def get_weight_jacobian(self, reg_params_free, time_w=None):
        """Return dKL / d reg_params_free d time_w.

        Parameters
        --------------
        reg_params_free : `numpy.ndarray`
            A flat free value for the regression parameters.
        time_w : `numpy.ndarray` (N,), optional
            A vector of time weights.  If not set, ``self.time_w`` is used.

        Returns
        -------------
        jac : `numpy.ndarray`
            The Jacobian of dKL / d reg_params_free with respect to the
            time weights.
        """
        if time_w is None:
            time_w = self.time_w
        jac = self.get_flat_time_kl_jacobian(reg_params_free, time_w)
        if self._reverse_jac_order:
            return jac.T
        else:
            return jac

    def to_json(self):
        """Return a JSON representation of the class.
        """
        reg_dict = dict()
        reg_dict['x_shape'] = self.x.shape
        reg_dict['y_shape'] = self.y.shape

        data_pattern = _get_data_pattern(self.x.shape, self.y.shape)
        data_flat = data_pattern.flatten(
            { 'x': self.x, 'y': self.y }, free=False)
        reg_dict['data_flat'] = list(data_flat)
        reg_dict['reverse_jac_order'] = self._reverse_jac_order
        return json.dumps(reg_dict)

    @classmethod
    def from_json(cls, json_str):
        """Instantiate a regression object from a json object.
        """

        reg_dict = json.loads(json_str)
        data_pattern = _get_data_pattern(
            tuple(reg_dict['x_shape']), tuple(reg_dict['y_shape']))
        data_dict = data_pattern.fold(reg_dict['data_flat'], free=False)
        regs = cls(y=data_dict['y'], x=data_dict['x'],
            reverse_jac_order=reg_dict['reverse_jac_order'])
        return regs
