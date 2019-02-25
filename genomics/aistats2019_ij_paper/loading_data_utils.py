"""Utilities for loading raw genomics data from the git repo
https://github.com/NelleV/genomic_time_series_bnp into a format that
can be used with our analyis.

Before using this scripts, you must:
1. Run ``make`` in ``genomic_time_series_bnp/data/``
2. Then run `make all` in
   ``genomic_time_series_bnp/src/exploratory_analysis/``.
"""

import os
import itertools

import numpy as np
import scipy as sp
import pandas as pd

import warnings

def _load_raw_data(
        condition="C",
        dataset_filename="data/shoemaker2015reprocessed/mice_data.txt",
        meta_filename="data/shoemaker2015reprocessed/mice_meta.txt",
        filter_de_genes_from=None, pval_thres=0.01,
        max_genes=None, return_meta=False):
    """
    Loads and pre-processes the raw genomics data.

    This is copied from
    genomic_time_series_bnp/src/splines_clustering/load_data.py

    Parameters
    ----------

    condition : string, default
        which condition to load.
        For shoemaker, can be C, K, M, VH, VL

    filter_de_genes_from: string
        Path to the differential expression files, containing the list of
        genes and the results of the DE analysis formatted as:

                      pval, lf
            NM_021274, 0, 2.42
            NM_021274, 0, 2.14

    pval_thres : float, optional, default: 0.01
        The p-value threshold.

    max_genes : int, optional, default: None
        The maximum number of genes to load. If provided, the top `max_genes`
        genes ranked in terms of p-values are returned.


    Returns
    -------
    timepoints, mapping, gene_names, X : tuple
        - timepoints is the ndarray (p, ) of timepoints
        - mapping is the ndarray (p, ) that maps uniquely each timepoints to a
          coordinate (used for the DE analysis)
        - gene_names : ndarray (n, ) of gene names.
        - X ndarray (n, p) of gene expression, where n is the number of genes
          returned, and p the number of sample measured

    """
    X = pd.read_csv(
        dataset_filename,
        delim_whitespace=True)
    meta = pd.read_csv(
        meta_filename,
        delim_whitespace=True)

    if "cho1998" in dataset_filename:
        X.set_index(X.columns[0], inplace=True)

    if type(condition) == list:
        which_to_keep = meta.loc[
            (np.isin(meta["condition"], condition))]
    else:
        which_to_keep = meta.loc[
            (meta["condition"] == condition)]

    X = X[which_to_keep["id"]]
    meta = which_to_keep
    gene_names = X.index
    timepoints = which_to_keep.time.as_matrix().astype(float)

    if filter_de_genes_from is not None:
        de_genes_exp = pd.read_csv(filter_de_genes_from)
        which_to_keep = de_genes_exp.loc[
            de_genes_exp["pval"] < pval_thres]
        if max_genes is not None:
            which_to_keep = which_to_keep.pval.argsort()[:max_genes]

        X = X.loc[which_to_keep.index].as_matrix().astype(float)
        gene_names = np.array(which_to_keep.index)
    else:
        X = X.as_matrix().astype(float)

        if max_genes is not None:
            X = X[:max_genes]
            gene_names = gene_names[:max_genes]

    order = timepoints.argsort()
    timepoints = timepoints[order]
    X = X[:, order]
    meta = meta.iloc[order]

    mapping = np.zeros(timepoints.shape)
    for c, i in enumerate(np.unique(timepoints)):
        mapping[timepoints == i] = c

    if not return_meta:
        return timepoints, mapping, gene_names, X
    else:
        return timepoints, mapping, gene_names, X, meta


def load_genomics_data(genomic_time_series_dir,
                       split_test_train=False,
                       train_indx_file=None):
    """Load the Shoemaker 2015 mice data in a form amenable to our analysis.

    Parameters
    ----------
    genomic_time_series_dir: `str`
        A local location of a clone of the git repo
        https://github.com/NelleV/genomic_time_series_bnp.  This function
        assumes that you have run ``make`` in the ``data`` directory of the
        repo to download and pre-process the raw dataset.
    split_test_train: `bool`, optional
        If true, split the data into training and testing observations using
        the indices in the file ``train_indx_file``.
    train_indx_file: `str`, optional
        The filename of a numpy save file containing 0-based indices of the
        training data.

    Returns
    -------
    y_train: `numpy.ndarray` (genes, timepoints)
        Training data in the form of an array of gene expression levels where
        the rows are genes and the columns are timepoints.
    y_test: `numpy.ndarray` (genes, timepoints)
        Test data in the same format as ``y_train``.  If ``split_test_train``
        is ``False``, this is ``None``.
    y_test: `numpy.ndarray` (genes, timepoints)
        Test data in the same format as ``y_train``.
        If ``split_test_train`` is ``False``, this is ``None``.
    train_indx: `numpy.ndarray` (genes, )
        The indices genes in the of the training set.
    timepoints: `numpy.ndarray` (N, )
        The timepoints of the gene expressions.
    """
    dataset_dir = os.path.join(
        genomic_time_series_dir, 'data/shoemaker2015reprocessed')
    print('Loading data from: ', dataset_dir)

    dataset_filename = 'mice_data.txt'
    metadata_filename = 'mice_meta.txt'

    filter_de_genes_from = os.path.join(
        genomic_time_series_dir, 'src/exploratory_analysis/results/M_vs_C.txt')

    max_genes = 1000

    condition = 'C'

    data_description_dict = dict(
        condition=condition,
        dataset_filename=os.path.join(dataset_dir, dataset_filename),
        meta_filename=os.path.join(dataset_dir, metadata_filename),
        filter_de_genes_from=filter_de_genes_from,
        max_genes=max_genes,
        return_meta=True)

    timepoints_float, mapping, gene_names, y, meta = \
        _load_raw_data(**data_description_dict)

    # We will need to be able to identify duplicated timpoints, so it is
    # bad form to compare floats.  As of now the analysis uses only integer
    # timepoints.
    timepoints = []
    for t in timepoints_float:
        t_int = int(t)
        if t_int != t:
            warnings.warn(
                'Non-integer timepoint {} being converted to integer.'.format(
                    t))
        timepoints.append(t_int)

    if split_test_train:
        assert train_indx_file is not None
        train_indx = np.load(train_indx_file)
        y_train = y[train_indx, :]

        n_obs = np.shape(y)[0]
        test_set = np.ones(n_obs, dtype = bool)
        test_set[train_indx] = False

        y_test = y[test_set, :]

        return y_train, y_test, train_indx, timepoints

    else:
        train_indx = np.arange(y.shape[0])
        return y, None, train_indx, timepoints
