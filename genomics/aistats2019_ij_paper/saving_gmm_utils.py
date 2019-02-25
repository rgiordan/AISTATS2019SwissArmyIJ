import numpy as np
import scipy as sp
from scipy import sparse

from datetime import datetime

import json
import json_tricks

import paragami

from . import regression_lib as reg_lib
from . import regression_mixture_lib as rm_lib
from . import sensitivity_lib as sens_lib

def coo_matrix_to_json(sp_mat):
    if not sp.sparse.isspmatrix_coo(sp_mat):
        raise ValueError('sp_mat must be a scipy.sparse.coo_matrix')
    sp_mat_dict = {
        'data': json_tricks.dumps(sp_mat.data),
        'row': json_tricks.dumps(sp_mat.row),
        'col': json_tricks.dumps(sp_mat.col),
        'shape': sp_mat._shape,
        'type': 'coo_matrix' }
    return json.dumps(sp_mat_dict)


# Convert the output of pack_csr_matrix back into a csr_matrix.
def coo_matrix_from_json(spdict_json):
    spd = json.loads(spdict_json)
    assert spd['type'] == 'coo_matrix'

    data = json_tricks.loads(spd['data'])
    row = json_tricks.loads(spd['row'])
    col = json_tricks.loads(spd['col'])
    return sp.sparse.coo_matrix(
        ( data, (row, col)), shape = spd['shape'])


def coo_matrix_to_dict(sp_mat, mat_name):
    if not sp.sparse.isspmatrix_coo(sp_mat):
        raise ValueError('sp_mat must be a scipy.sparse.coo_matrix')
    sp_mat_dict = {
        mat_name + '_data': sp_mat.data,
        mat_name + '_row': sp_mat.row,
        mat_name + '_col': sp_mat.col,
        mat_name + '_shape': sp_mat._shape }
    return sp_mat_dict

# Convert the output of pack_csr_matrix back into a csr_matrix.
def coo_matrix_from_dict(spdict, mat_name):
    data = spdict[mat_name + '_data']
    row = spdict[mat_name + '_row']
    col = spdict[mat_name + '_col']
    mat_shape = spdict[mat_name + '_shape']
    return sp.sparse.coo_matrix(
        ( data, (row, col)), shape=mat_shape)

def get_result_metadata(timepoints, description=''):
    """A dictionary containing extra information about the analysis.
    """
    metadata = dict()
    metadata['timepoints'] = timepoints
    metadata['description'] = description
    metadata['datetime'] = str(datetime.today().timestamp())
    return metadata


def get_initial_fit_filename(
        df, degree, num_components, base_name='mice_data'):
    return \
        '{}_initial_fit_df{}_degree{}_num_components{}.npz'.format(
            base_name, df, degree, num_components)

def get_test_regs_filename(df, degree, base_name='mice_data'):
    return \
        '{}_test_reg_df{}_degree{}.json'.format(base_name, df, degree)

def get_refit_filename(df, degree, num_components,
                       lo_num_times, lo_which_comb, lo_max_num_timepoints,
                       init_method):
    return \
        ('mice_data_refit_' +
         'nt{}_comb{}_maxt{}_' +
         'df{}_degree{}_num_components{}_init{}.npz').format(
            lo_num_times,  lo_which_comb, lo_max_num_timepoints,
            df, degree, num_components, init_method)


def save_initial_optimum(outfile, gmm, regs, fit_derivs,
                         timepoints, extra_metadata=None):
    metadata = get_result_metadata(timepoints)
    if extra_metadata is not None:
        metadata.update(extra_metadata)

    full_estimand_dict = fit_derivs.to_numpy_dict()
    full_estimand_dict['regs_json'] = regs.to_json()
    full_estimand_dict['gmm_json'] = gmm.to_json()
    full_estimand_dict['metadata_json'] = json_tricks.dumps(metadata)
    np.savez_compressed(outfile, **full_estimand_dict)


def load_initial_optimum(infile):
    with np.load(infile) as res:
        fit_derivs = sens_lib.FitDerivatives.from_numpy_dict(res)
        comb_params = fit_derivs.get_comb_params()
        regs = reg_lib.Regressions.from_json(str(res['regs_json']))
        gmm = rm_lib.GMM.from_json(
            str(res['gmm_json']), regs, comb_params['reg'])
        metadata = json_tricks.loads(str(res['metadata_json']))

    return fit_derivs, gmm, regs, metadata


def load_initial_optimum_fit_only(infile):
    with np.load(infile) as res:
        comb_params_pattern = paragami.get_pattern_from_json(
            str(res['comb_params_pattern_json']))
        comb_params_free = res['comb_params_free']
        comb_params = comb_params_pattern.fold(comb_params_free, free=True)
        regs = reg_lib.Regressions.from_json(str(res['regs_json']))
        gmm = rm_lib.GMM.from_json(
            str(res['gmm_json']), regs, comb_params['reg'])
        metadata = json_tricks.loads(str(res['metadata_json']))

    return comb_params_pattern, comb_params, gmm, regs, metadata


def save_refit(outfile,
               comb_params_free,
               comb_params_pattern,
               initial_fit_infile,
               time_w,
               lo_inds,
               full_lo_inds,
               extra_metadata=None):

    metadata = dict()
    metadata['time_w'] = time_w
    metadata['lo_inds'] = lo_inds
    metadata['full_lo_inds'] = full_lo_inds
    if extra_metadata is not None:
        metadata.update(extra_metadata)

    refit_dict = dict()
    refit_dict['comb_params_free'] = comb_params_free
    refit_dict['comb_params_pattern_json'] = comb_params_pattern.to_json()
    refit_dict['initial_fit_infile'] = initial_fit_infile
    refit_dict['metadata_json'] = json_tricks.dumps(metadata)
    np.savez_compressed(outfile, **refit_dict)


def load_refit(infile):
    with np.load(infile) as res:
        comb_params_free = res['comb_params_free']
        comb_params_pattern = paragami.PatternDict.from_json(
            str(res['comb_params_pattern_json']))
        metadata = json_tricks.loads(str(res['metadata_json']))

    return comb_params_free, comb_params_pattern, metadata


def get_prediction_error_filename(
        df, degree, num_components, lo_num_times, init_method,
        lo_max_num_timepoints):
    """Get a default filename for the file with prediction errors.
    """

    return ('lo_prediction_error_df{df}_degree{degree}_' +
            '_num_components{num_components}_nt{lo_num_times}_' +
            'maxt{lo_max_num_timepoints}_init{init_method}.npz').format(
                df=df,
                degree=degree,
                num_components=num_components,
                lo_num_times=lo_num_times,
                init_method=init_method,
                lo_max_num_timepoints=lo_max_num_timepoints)
