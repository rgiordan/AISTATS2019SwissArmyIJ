#!/usr/bin/env python3
"""Create shell scripts to run ``refit.py`` for a range of parameters.
Optionally submit the shell scripts to slurm.  This depends on the output
of ``run_slurm_initial_fits.py``.

Example usage:
./run_slurm_refits.sh \
    --lo_num_times 1
"""

import argparse
import os
import cluster_scripts_lib as util
import subprocess
import scipy as sp
from scipy import special

parser = argparse.ArgumentParser()
parser.add_argument('--outfolder', default=None, type=str)
parser.add_argument('--submit', dest='submit', action='store_true',
                    help='Submit to slurm.')
parser.add_argument('--no-submit', dest='submit', action='store_false',
                    help='Do not submit to slurm.')
parser.add_argument('--lo_num_times', type=int, required=True,
                    help='How many times to leave out.')
parser.set_defaults(submit=False)

args = parser.parse_args()

if args.outfolder is None:
    if args.submit:
        args.outfolder = \
            '/scratch/users/genomic_times_series_bnp/aistats_results/'
    else:
        args.outfolder = '../fits/'

# refit.py expects the input and output to be in the same folder.
if not os.path.isdir(args.outfolder):
    raise ValueError('Destination directory {} does not exist.'.format(
        args.outfolder))

lo_max_num_timepoints = 7
lo_num_times = args.lo_num_times
num_comb = int(sp.special.comb(lo_max_num_timepoints, lo_num_times))

if num_comb > 50:
    raise ValueError('Warning: num_comb > 50. num_comb = {}'.format(num_comb))

for df in util.df_range:
    for init_method in ['warm', 'kmeans']:
        for lo_which_comb in range(num_comb):
            script_name = ('mice_data_refit_' +
             'nt{}_comb{}_maxt{}_df{}_init{}.sh').format(
                lo_num_times,  lo_which_comb, lo_max_num_timepoints,
                df, init_method)
            full_script_name = os.path.join(util.slurm_dir, script_name)
            with open(full_script_name, 'w') as slurm_script:
                slurm_script.write('#!/bin/bash\n')
                slurm_script.write(util.activate_venv_cmd)
                cmd = ('{path}/refit.py ' +
                       '--outfolder {outfolder} ' +
                       '--df {df} ' +
                       '--init_method {init_method} ' +
                       '--lo_num_times {lo_num_times} ' +
                       '--lo_which_comb {lo_which_comb} ' +
                       '--lo_max_num_timepoints {lo_max_num_timepoints} ' +
                       '\n').format(
                    outfolder=args.outfolder,
                    path=util.script_dir,
                    df=df,
                    init_method=init_method,
                    lo_num_times=lo_num_times,
                    lo_which_comb=lo_which_comb,
                    lo_max_num_timepoints=lo_max_num_timepoints)
                slurm_script.write(cmd)
            if args.submit:
                print('Submitting {}'.format(full_script_name))
                command = ['sbatch', full_script_name]
                subprocess.run(command)
            else:
                print('Generating (but not submitting) shell script {}'.format(
                    full_script_name))
