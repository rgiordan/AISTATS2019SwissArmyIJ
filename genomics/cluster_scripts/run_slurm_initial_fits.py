#!/usr/bin/env python3
"""Create shell scripts to run ``initial_fit.py`` for a range of parameters.
Optionally submit the shell scripts to slurm.
"""

import argparse
import os
import cluster_scripts_lib as util
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--outfolder', default=None, type=str)
parser.add_argument('--submit', dest='submit', action='store_true',
                    help='Submit to slurm.')
parser.add_argument('--no-submit', dest='submit', action='store_false',
                    help='Submit to slurm.')
parser.set_defaults(submit=False)

args = parser.parse_args()

if args.outfolder is None:
    if args.submit:
        args.outfolder = \
            '/scratch/users/genomic_times_series_bnp/aistats_results/'
    else:
        args.outfolder = '../fits/'

if not os.path.isdir(args.outfolder):
    raise ValueError('Destination directory {} does not exist.'.format(
        args.outfolder))

for df in util.df_range:
    script_name = 'initial_fit_df{}.sh'.format(df)
    full_script_name = os.path.join(util.slurm_dir, script_name)
    with open(full_script_name, 'w') as slurm_script:
        slurm_script.write('#!/bin/bash\n')
        slurm_script.write(util.activate_venv_cmd)
        slurm_script.write(
            '{}/initial_fit.py --df {} --outfolder {}\n'.format(
            util.script_dir, df, args.outfolder))
    if args.submit:
        print('Submitting {}'.format(full_script_name))
        command = ['sbatch', full_script_name]
        subprocess.run(command)
    else:
        print('Generating (but not submitting) shell script {}'.format(
            full_script_name))
