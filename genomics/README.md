# Contents

This folder contains scripts for actually running the analyses reported in
the paper.  The steps are

1. Install required packages.
1. Setup and run the tests.
1. Get and preprocess the data.
1. Run the analysis.
    1. Run the initial fits (``initial_fit.py`` and ``run_slurm_initial_fits.py``).
    2. Run cross-validation (``refit.py`` and ``run_slurm_refits.py``).
    3. Calcualte the IJ estimates and prediciton errors
       (``calculate_prediction_errors.py`` and ``run_slurm_pred_error.py``).
1. Analyze the results (see the ``../jupyter`` directory).


## Install Required Packages.

The easiest way to do this is with a virtual environment and ``pip``.  In
the root of this repository, run

```
virtualenv -p /usr/bin/python3 --no-site-packages venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install .
```

## Run the tests.

To make sure everything has been installed correctly, you
can run the tests.  First, you must generate test files with the
script ``generate_test_fit.py``.  This needs to be done only once.

```
aistats2019_ij_paper/tests/generate_test_fit.py
python3 -m unittest discover -s aistats2019_ij_paper/tests
```

## Get and preprocess the data.

To get the genomics data, you need to clone the git repository
[NelleV/genomic_time_series_bnp](https://github.com/NelleV/genomic_time_series_bnp).
Then run ``make`` in two locations:

1. Run ``make`` in ``genomic_time_series_bnp/data/``
2. Then run `make all` in
   ``genomic_time_series_bnp/src/exploratory_analysis/``.

## Run the analysis.

The scripts to analyze the data are in the folder ``cluster_scripts``.
There are three steps:

1. Run the shell scripts created with ``run_slurm_initial_fits.py``.
1. Then, run the shell scripts created with ``run_slurm_refits.py``.
1. Then, run the shell scripts created with
    1. ``run_slurm_initial_fits.py --lo_num_times 1``
    1. ``run_slurm_initial_fits.py --lo_num_times 2``
    1. ``run_slurm_initial_fits.py --lo_num_times 3``

By default, each script only generates shell files to run the corresponding
analyses, but if the ``--submit`` flag is set, it will also submit these
shell files to slurm using ``sbatch``.

By default, the files are saved in the ``fits`` directory
``--submit`` is set, in which case they are saved by default to
``/scratch/users/genomic_times_series_bnp/aistats_results``.

## Analyze the results.

Scripts for interactively running the analysis, including examining
the final prediction error results and producing the data
for the paper are contained in the ``jupyter`` folder.  See
``jupyter/README.md`` for information on setting up a virtual environment
for the notebooks.

In particular, the data for the paper is produced by the notebook
``jupyter/R/examine_and_save_results.ipynb``.  After the analysis above has
been completed, you can run:

```
cd jupyter/R
jupyter nbconvert --to notebook --execute examine_and_save_results.ipynb \
    --output /tmp/examine_prediction_errors.ipynb
```

A side effect of this command should be an Rdata file in the ``fits`` folder
that can be used to generate the paper's graphs.
