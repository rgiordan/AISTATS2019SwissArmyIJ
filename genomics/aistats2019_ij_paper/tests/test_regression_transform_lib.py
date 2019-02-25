#!/usr/bin/env python3

import aistats2019_ij_paper.transform_regression_lib as trans_reg_lib
import numpy as np
from numpy.testing import assert_array_almost_equal
import unittest


class TestTransforms(unittest.TestCase):
    def test_make_matrix_full_row_rank_with_unrotation(self):
        # x would be the regressors in actual usage.
        x = np.random.random((42, 7))

        x_new, x_unrotate = \
            trans_reg_lib.make_matrix_full_row_rank_with_unrotation(x)

        assert_array_almost_equal(x_new.T @ x_new, x.T @ x)
        v = np.random.random(x.shape[1])
        assert_array_almost_equal(x @ v, x_unrotate @ x_new @ v)

    def test_get_reversible_predict_and_demean_matrix(self):
        x = np.random.random((42, 7))
        t_mat, ut_mat = \
            trans_reg_lib.get_reversible_predict_and_demean_matrix(x)

        beta = np.random.random(x.shape[1])
        pred_y = x @ beta - np.mean(x @ beta)
        gamma = t_mat @ beta

        assert_array_almost_equal(
            np.linalg.norm(gamma), np.linalg.norm(pred_y))
        assert_array_almost_equal(ut_mat @ gamma, pred_y)


if __name__ == '__main__':
    unittest.main()
