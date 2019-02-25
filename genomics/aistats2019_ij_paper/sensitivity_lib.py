import autograd
import autograd.numpy as np
import autograd.scipy as sp
from copy import deepcopy
import json
import paragami
from . import saving_gmm_utils
import scipy as osp
import time

def get_combined_parameters_pattern(reg_params_pattern, gmm_params_pattern):
    """A paragami pattern for the combined regression parameters (`reg`)
    and an optimal clustering (`mix`).
    """
    comb_params_pattern = paragami.PatternDict()
    comb_params_pattern['reg'] = reg_params_pattern
    comb_params_pattern['mix'] = gmm_params_pattern
    comb_params_pattern.lock()
    return comb_params_pattern


class FitDerivatives(object):
    """Summarizes the regression and clustering together.

    The key attributes are ``full_hess`` and ``t_jac``, which can be
    passed to a hyperparameter sensitivity class.

    This class saves a joint optimum together with the "Hessian" and weight
    sensitivity Jacobian.
    """
    def __init__(self,
                 opt_gmm_params, opt_reg_params,
                 gmm_params_pattern, reg_params_pattern,
                 gmm=None, regs=None,
                 full_hess=None, t_jac=None,
                 verbose=True, print_every=0):
        """
        Parameters
        ------------
        opt_gmm_params :
            The mixture parameters at the optimum.
        opt_reg_params :
            The regression parameters at the optimum.
        gmm_params_pattern : `paragami.PatternDict`
            The pattern for the mixture parameters.
        reg_params_pattern : `paragami.PatternDict`
            The pattern for the regression parameters.
        gmm : `GMM`, optional
            A GMM fit.  This can be used to calculate the Hessian and Jacobian
            if they are not provided.
        regs : `Regressions`, optional
            A Regressions fit.  This can be used to calculate the Hessian
            and Jacobian if they are not provided.
        full_hess : matrix
            The derivative of the full set of estimating equations, i.e.,
            the "Hessian".
        t_jac : matrix
            The derivative of the estimating equations with respect to
            the time weights.
        """
        self.verbose = verbose

        if gmm is not None:
            if gmm.gmm_params_pattern != gmm_params_pattern:
                raise ValueError('The gmm patterns do not match.')

        if regs is not None:
            if regs.reg_params_pattern != reg_params_pattern:
                raise ValueError('The regs patterns do not match.')

        self.comb_params_pattern = \
            get_combined_parameters_pattern(
                reg_params_pattern, gmm_params_pattern)

        self.reg_len = self.comb_params_pattern['reg'].flat_length(free=True)
        self.mix_len = self.comb_params_pattern['mix'].flat_length(free=True)
        self.comb_len = self.reg_len + self.mix_len

        self.initialize(opt_gmm_params, opt_reg_params,
                        gmm=gmm, regs=regs,
                        full_hess=full_hess, t_jac=t_jac,
                        print_every=print_every)

        self.reg_inds = self.comb_params_pattern.flat_indices({
            'reg': reg_params_pattern.empty_bool(True),
            'mix': gmm_params_pattern.empty_bool(False) }, free=True)

        self.gmm_inds = self.comb_params_pattern.flat_indices({
            'reg': reg_params_pattern.empty_bool(False),
            'mix': gmm_params_pattern.empty_bool(True) }, free=True)

        # Make sure the packing is as expected.  This is necessary
        # because we use hstack to combine the Hessians.
        if np.any(self.reg_inds != np.arange(self.reg_len)):
            raise ValueError('Wrong pattern for reg_inds')
        if np.any(self.gmm_inds !=
                  np.arange(self.reg_len, self.comb_len)):
            raise ValueError('Wrong pattern for gmm_inds')

    def verbose_print(self, s):
        if self.verbose:
            print(s)

    def initialize(self, opt_gmm_params, opt_reg_params,
                   gmm=None, regs=None,
                   full_hess=None, t_jac=None, print_every=0):
        if (gmm is None) or (regs is None):
            if (full_hess is None) or (t_jac is None):
                raise ValueError((
                    'If ``gmm`` and ``regs`` are not specified, you must ' +
                    'specify ``full_hess`` and ``t_jac``.'))
        self.verbose_print('Initializing FitDerivatives.')
        self.gmm_params_free = \
            self.comb_params_pattern['mix'].flatten(opt_gmm_params, free=True)
        self.reg_params_free = \
            self.comb_params_pattern['reg'].flatten(opt_reg_params, free=True)
        comb_params = { 'reg': opt_reg_params, 'mix': opt_gmm_params }
        self.comb_params_free = self.comb_params_pattern.flatten(
            comb_params, free=True)

        self.gmm_params_free.setflags(write=False)
        self.reg_params_free.setflags(write=False)
        self.comb_params_free.setflags(write=False)

        if t_jac is not None:
            self.verbose_print('Using provided t_jac.')
            self.t_jac = t_jac
        else:
            self.verbose_print('Getting t Jacobian.')
            self.t_jac = self.get_t_weight_jacobian(
                regs, self.reg_params_free)

        if full_hess is not None:
            self.verbose_print('Using provided full_hess.')
            self.full_hess = full_hess
        else:
            self.verbose_print('Getting full Hessian.')
            self.full_hess = self.get_two_stage_hessian(
                gmm, regs, self.gmm_params_free, self.reg_params_free,
                print_every=print_every)
        if self.full_hess.shape != (self.comb_len, self.comb_len):
            raise ValueError('full_hess is the wrong shape.')

    def get_comb_params(self):
        return self.comb_params_pattern.fold(self.comb_params_free, free=True)

    def get_t_weight_jacobian(self, regs, reg_params_free):
        """Calculate the "Jacobian" of the two-stage estimator -- that is,
        the derivative of the estimating equation with respect to the weights.
        """
        w_t_reg_jac = regs.get_weight_jacobian(reg_params_free)
        num_w_t_pars = w_t_reg_jac.shape[0]

        # The GMM does not depend directly on the time weights, so the
        # Jacobian is zero for those parameters.
        w_t_gmm_jac = osp.sparse.coo_matrix((self.mix_len, num_w_t_pars))
        t_jac = osp.sparse.vstack([w_t_reg_jac.T, w_t_gmm_jac])
        return t_jac

    def get_two_stage_hessian(self, gmm, regs, gmm_params_free, reg_params_free,
                              print_every=0):
        """Calculate the "Hessian" of the two-stage estimator -- that is,
        the derivative of the estimating equation.
        """
        gmm_hess_time = time.time()
        self.verbose_print('   Getting GMM Hessian...')
        self.gmm_hess = gmm.kl_obj.hessian(gmm_params_free)
        gmm_hess_time = time.time() - gmm_hess_time
        self.verbose_print('   GMM Hessian time: {}'.format(gmm_hess_time))

        cross_hess_time = time.time()
        self.verbose_print('   Getting cross Hessian...')
        self.cross_hess = \
            gmm.get_reg_params_kl_flat_cross_hess(
                gmm_params_free, reg_params_free)
        cross_hess_time = time.time() - cross_hess_time
        self.verbose_print(
            '   Cross Hessian time: {}'.format(cross_hess_time))

        reg_hess_time = time.time()
        self.verbose_print('   Getting regression Hessian...')
        self.reg_hess = regs.get_sparse_free_hessian(
            self.reg_params_free, print_every=print_every)
        reg_hess_time = time.time() - reg_hess_time
        self.verbose_print(
            '   Regression Hessian time: {}'.format(reg_hess_time))

        # The other cross matrix is zeros.
        self.cross_zeros = osp.sparse.coo_matrix(self.cross_hess.T.shape)

        full_hess = osp.sparse.vstack([
            osp.sparse.hstack([self.reg_hess, self.cross_zeros]),
            osp.sparse.hstack([osp.sparse.coo_matrix(self.cross_hess),
                               osp.sparse.coo_matrix(self.gmm_hess)])])
        self.verbose_print('Done with full Hessian.')

        return full_hess

    def to_json(self):
        """Because json does no compression, it is better to use
        ``to_numpy_dict``.
        """
        pred_dict = dict()
        pred_dict['comb_params_pattern_json'] = \
            self.comb_params_pattern.to_json()
        pred_dict['comb_params_free'] = list(self.comb_params_free)
        pred_dict['t_jac_json'] = \
            saving_gmm_utils.coo_matrix_to_json(self.t_jac)
        pred_dict['full_hess_json'] = \
            saving_gmm_utils.coo_matrix_to_json(self.full_hess)
        pred_dict['verbose'] = self.verbose
        return json.dumps(pred_dict)

    @classmethod
    def from_json(cls, json_string):
        """Because json does no compression, it is better to use
        ``from_numpy_dict``.
        """
        pred_dict = json.loads(json_string)
        comb_params_pattern = paragami.PatternDict.from_json(
            pred_dict['comb_params_pattern_json'])
        comb_params = comb_params_pattern.fold(
            pred_dict['comb_params_free'], free=True)
        t_jac = saving_gmm_utils.coo_matrix_from_json(
            pred_dict['t_jac_json'])
        full_hess = saving_gmm_utils.coo_matrix_from_json(
            pred_dict['full_hess_json'])

        return  cls(
            opt_gmm_params=comb_params['mix'],
            opt_reg_params=comb_params['reg'],
            gmm_params_pattern=comb_params_pattern['mix'],
            reg_params_pattern=comb_params_pattern['reg'],
            gmm=None,
            full_hess=full_hess,
            t_jac=t_jac,
            verbose=pred_dict['verbose'],
            print_every=0)

    def to_numpy_dict(self):
        """Return a dictionary of numpy arrays that can be passed to
        ``numpy.savez_compressed`` as keyword arguments.  This can save
        a lot of disk space relative to json.

        Example:

        >>> full_estimand_dict = full_estimand.to_numpy_dict()
        >>> np.savez_compressed('savefile.npz', **full_estimand_dict)
        >>> with np.load('savefile.npz') as res:
        ...     full_estimand_dict2 = FullEstimand.from_numpy_dict(res)

        Then ``full_estimand_dict2`` and ``full_estimand_dict`` will be
        effectively equivalent.
        """
        np_dict = dict()
        np_dict['comb_params_pattern_json'] = \
            self.comb_params_pattern.to_json()
        np_dict['comb_params_free'] = self.comb_params_free
        np_dict['verbose'] = self.verbose
        np_dict.update(
            saving_gmm_utils.coo_matrix_to_dict(
                self.t_jac, mat_name='t_jac'))
        np_dict.update(
            saving_gmm_utils.coo_matrix_to_dict(
                self.full_hess, mat_name='full_hess'))
        return np_dict

    @classmethod
    def from_numpy_dict(cls, numpy_dict):
        full_hess = saving_gmm_utils.coo_matrix_from_dict(
            numpy_dict, 'full_hess')
        t_jac = saving_gmm_utils.coo_matrix_from_dict(
            numpy_dict, 't_jac')
        comb_params_pattern = paragami.get_pattern_from_json(
            str(numpy_dict['comb_params_pattern_json']))
        comb_params_free = numpy_dict['comb_params_free']
        comb_params = comb_params_pattern.fold(
            numpy_dict['comb_params_free'], free=True)
        verbose = bool(numpy_dict['verbose'])
        return  cls(
            opt_gmm_params=comb_params['mix'],
            opt_reg_params=comb_params['reg'],
            gmm_params_pattern=comb_params_pattern['mix'],
            reg_params_pattern=comb_params_pattern['reg'],
            gmm=None,
            full_hess=full_hess,
            t_jac=t_jac,
            verbose=verbose,
            print_every=0)
