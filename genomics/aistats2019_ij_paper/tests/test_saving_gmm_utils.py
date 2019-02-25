#!/usr/bin/env python3

import unittest

import os
import numpy as np
from numpy.testing import assert_array_almost_equal
import scipy
from scipy import sparse

from aistats2019_ij_paper import saving_gmm_utils
from aistats2019_ij_paper import mse_utils
from aistats2019_ij_paper import loading_data_utils

from aistats2019_ij_paper.tests import utils_for_testing_lib
from aistats2019_ij_paper.tests import generate_test_fit

class TestUtils(unittest.TestCase):
    def test_coo_matrix_to_json(self):
        import scipy

        foo = np.random.random((3, 2))
        foo[0, 1] = 0
        foo[2, 0] = 0
        foo_sp = scipy.sparse.coo_matrix(foo)

        foo_sp_json = saving_gmm_utils.coo_matrix_to_json(foo_sp)
        foo_sp_new = saving_gmm_utils.coo_matrix_from_json(foo_sp_json)

        assert_array_almost_equal(
            foo_sp_new.todense(), foo_sp.todense())

    def test_save_refit(self):
        fit_derivs, gmm, regs, metadata = generate_test_fit.load_test_derivs()
        new_time_w = np.random.random(len(regs.time_w))

        timepoints = metadata['timepoints']
        lo_inds = mse_utils.get_indexed_combination(
            num_times=2, which_comb=3, max_num_timepoints=7)
        new_time_w, full_lo_inds = \
            mse_utils.get_time_weight(lo_inds, timepoints)

        initial_fit_infile = 'abcd'
        save_filename = os.path.join(
            generate_test_fit.test_data_dir, 'refit_test.npz')
        saving_gmm_utils.save_refit(
            outfile=save_filename,
            comb_params_free=fit_derivs.comb_params_free,
            comb_params_pattern=fit_derivs.comb_params_pattern,
            initial_fit_infile=initial_fit_infile,
            time_w=new_time_w,
            lo_inds=lo_inds,
            full_lo_inds=full_lo_inds)

        comb_params_free1, comb_params_pattern1, metadata1 = \
            saving_gmm_utils.load_refit(save_filename)
        assert_array_almost_equal(
            fit_derivs.comb_params_free, comb_params_free1)
        self.assertEqual(
            fit_derivs.comb_params_pattern,
            comb_params_pattern1)

        assert_array_almost_equal(new_time_w, metadata1['time_w'])
        assert_array_almost_equal(lo_inds, metadata1['lo_inds'])
        assert_array_almost_equal(full_lo_inds, metadata1['full_lo_inds'])


    def test_save_initial_optimum(self):
        fit_derivs, gmm, regs, metadata = generate_test_fit.load_test_derivs()
        extra_metadata = dict()
        extra_metadata['xiao'] = 'wang'
        timepoints = metadata['timepoints']

        npz_outfile = os.path.join(
            generate_test_fit.test_data_dir, 'initial_fit_test.npz')
        saving_gmm_utils.save_initial_optimum(
            npz_outfile,
            gmm=gmm,
            regs=regs,
            timepoints=timepoints,
            fit_derivs=fit_derivs,
            extra_metadata=extra_metadata)

        fit_derivs1, gmm1, regs1, metadata1 = \
            saving_gmm_utils.load_initial_optimum(npz_outfile)

        utils_for_testing_lib.assert_gmm_equal(self, gmm, gmm1)
        utils_for_testing_lib.assert_regressions_equal(self, regs, regs1)
        utils_for_testing_lib.assert_fit_derivs_equal(
            self, fit_derivs, fit_derivs1)
        assert_array_almost_equal(timepoints, metadata1['timepoints'])
        self.assertEqual('wang', metadata1['xiao'])

        comb_params_pattern2, comb_params2, gmm2, regs2, metadata1 = \
            saving_gmm_utils.load_initial_optimum_fit_only(npz_outfile)
        utils_for_testing_lib.assert_gmm_equal(self, gmm, gmm2)
        utils_for_testing_lib.assert_regressions_equal(self, regs, regs2)
        self.assertEqual(fit_derivs.comb_params_pattern, comb_params_pattern2)
        assert_array_almost_equal(
            fit_derivs.comb_params_free,
            comb_params_pattern2.flatten(comb_params2, free=True))
        self.assertEqual('wang', metadata1['xiao'])



if __name__ == '__main__':
    unittest.main()
