#!/usr/bin/env python3

import copy
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

import paragami

from aistats2019_ij_paper import regression_lib as reg_lib
from aistats2019_ij_paper import regression_mixture_lib as rm_lib
from  aistats2019_ij_paper import saving_gmm_utils
from aistats2019_ij_paper import mse_utils

from aistats2019_ij_paper.tests import generate_test_fit

class TestMSE(unittest.TestCase):
    def test_prediction(self):
        fit_derivs, gmm, regs, metadata = generate_test_fit.load_test_derivs()

        opt_comb_params = fit_derivs.get_comb_params()
        y_pred = mse_utils.get_predictions(
            gmm, opt_comb_params['mix'], opt_comb_params['reg'])
        self.assertEqual(regs.y.shape, y_pred.shape)

    def test_timepoints_to_int(self):
        timepoints_float = [1.0, 3.0, 5.0, 5.0]
        timepoints_int = mse_utils.timepoint_to_int(timepoints_float)
        assert_array_almost_equal(
            timepoints_float, timepoints_int)

    def test_get_time_weight(self):
        timepoints_float = [5, 1.0, 3, 3.0, 5.0, 5.0]
        def getw(lo_inds):
            return mse_utils.get_time_weight(lo_inds, timepoints_float)[0]

        def check_weights(expected, lo_inds):
            w, inds = mse_utils.get_time_weight(lo_inds, timepoints_float)
            assert_array_almost_equal(expected, w)
            if len(inds) > 0:
                self.assertTrue(np.all(np.array(expected)[inds] == 0))
            non_inds = np.setdiff1d(np.arange(len(w)), inds)
            if len(non_inds) > 0:
                self.assertTrue(np.all(np.array(expected)[non_inds] == 1))

        check_weights([1, 1, 1, 1, 1, 1], [])
        check_weights([1, 0, 1, 1, 1, 1], [0])
        check_weights([1, 1, 0, 0, 1, 1], [1])
        check_weights([0, 1, 1, 1, 0, 0], [2])
        check_weights([1, 0, 0, 0, 1, 1], [0, 1])
        check_weights([0, 0, 1, 1, 0, 0], [0, 2])
        check_weights([0, 1, 0, 0, 0, 0], [1, 2])
        check_weights([0, 0, 0, 0, 0, 0], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
