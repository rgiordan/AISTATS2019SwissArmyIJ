#!/usr/bin/env python3
"""Create shell scripts to run ``calculate_prediction_errors.py`` for a range
of parameters.  Optionally submit the shell scripts to slurm.
This depends on the output of ``run_slurm_initial_fits.py`` and
``run_slurm_refits.py``.

Example usage:
./run_slurm_pred_error.sh \
    --df 4 \
    --init_method warm \
    --lo_num_times 1 \
"""

import argparse
import os
import cluster_scripts_lib as util
import itertools
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--outfolder', default=None, type=str)
parser.add_argument('--submit', dest='submit', action='store_true',
                    help='Submit to slurm.')
parser.add_argument('--no-submit', dest='submit', action='store_false',
                    help='Do not submit to slurm.')
parser.set_defaults(submit=False)

args = parser.parse_args()

if args.outfolder is None:
    if args.submit:
        args.outfolder = \
            '/scratch/users/genomic_times_series_bnp/aistats_results/'
    else:
        args.outfolder = '../fits/'

# The script expects the input and output to be in the same folder.
if not os.path.isdir(args.outfolder):
    raise ValueError('Destination directory {} does not exist.'.format(
        args.outfolder))

for df, lo_num_times, init_method in itertools.product(
        util.df_range, util.lo_num_times_range, ['warm', 'kmeans']):
    script_name = ('mice_data_pred_error_' +
        'nt{}_df{}_init{}.sh').format(
            lo_num_times, df, init_method)
    full_script_name = os.path.join(util.slurm_dir, script_name)
    with open(full_script_name, 'w') as slurm_script:
        slurm_script.write('#!/bin/bash\n')
        slurm_script.write(util.activate_venv_cmd)
        cmd = ('{path}/calculate_prediction_errors.py ' +
               '--outfolder {outfolder} ' +
               '--df {df} ' +
               '--init_method {init_method} ' +
               '--lo_num_times {lo_num_times} ' +
               '\n').format(
            outfolder=args.outfolder,
            path=util.script_dir,
            df=df,
            init_method=init_method,
            lo_num_times=lo_num_times)
        slurm_script.write(cmd)
    if args.submit:
        print('Submitting {}'.format(full_script_name))
        command = ['sbatch', full_script_name]
        subprocess.run(command)
    else:
        print('Generating (but not submitting) shell script {}'.format(
            full_script_name))
