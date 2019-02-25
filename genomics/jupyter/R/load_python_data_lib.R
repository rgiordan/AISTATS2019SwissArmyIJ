# Load data saved with calculate_prediction-errors.py into R.
library(reticulate)
library(dplyr)
library(reshape2)

# By using the virtual environment python, we can access the
# paper's python packages.
PYTHON_BIN <- "../../venv/bin/python3"
stopifnot(file.exists(PYTHON_BIN))
use_python(PYTHON_BIN)

# Useful for constructing readable python commands using R variables.
`%_%` <- function(x, y) { paste(x, y, sep="")}


InitializePython <- function() {
    py_main <- reticulate::import_main()
    py_run_string("
import sys
import numpy as np
import os
from aistats2019_ij_paper import saving_gmm_utils
    ")
    return(py_main)
}


GetResultDataframe <- function(name) {
    # Assumes that load_filename is already defined in py_main.
    load_string <- "
with np.load(os.path.join('../../fits/', load_filename)) as infile:
    num_comb = infile['num_comb']
    total_lr_time = infile['lr_time']
    total_refit_time = infile['refit_time']
    initial_opt_time = infile['opt_time']
    initial_reg_time = infile['reg_time']
    initial_hess_time = infile['hess_time']
    gmm_param_length = infile['gmm_param_length']
    reg_param_length = infile['reg_param_length']

    infile_names = infile.keys()

    error_mat = infile[name + '_error_mat']
    orig_error_mat = infile[name + '_orig_error_mat']
    test_arr = infile[name + '_test_arr']
    method_arr = infile[name + '_method_arr']
    full_lo_inds_mat = infile[name + '_full_lo_inds_mat']
    comb_arr = infile[name + '_comb_arr']
    rereg_arr = infile[name + '_rereg_arr']
    row_arr = infile[name + '_row_arr']"

    py_run_string("name = '" %_% name %_%"'")
    py_run_string(load_string)

    err_df <- data.frame(py_main$error_mat)
    names(err_df) <- paste("err", 1:ncol(err_df), sep="")

    orig_err_df <- data.frame(py_main$orig_error_mat)
    names(orig_err_df) <- paste("train_err", 1:ncol(err_df), sep="")

    full_lo_inds <- data.frame(py_main$full_lo_inds_mat)
    names(full_lo_inds) <- paste("time", 1:ncol(full_lo_inds), sep="")
    err_df <- bind_cols(err_df, orig_err_df, full_lo_inds)

    err_df$test <- py_main$test_arr
    err_df$method <- py_main$method_arr
    err_df$comb <- py_main$comb_arr
    err_df$rereg <- py_main$rereg_arr
    err_df$gene <- py_main$row_arr

    return(err_df)
}


LoadPredictionError <- function(df, lo_num_times, init_method) {
    py_run_string("
df = " %_% df %_% "
degree = 3
num_components = 18
lo_num_times = " %_% lo_num_times %_% "
init_method = '" %_% init_method %_% "'
lo_max_num_timepoints = 7

load_filename = saving_gmm_utils.get_prediction_error_filename(
    df, degree, num_components, lo_num_times,
    init_method, lo_max_num_timepoints)
    ")

    refit_err_df <- GetResultDataframe("refits")
    refit_err_df$df <- df
    refit_err_df$lo_num_times <- lo_num_times
    refit_err_df$init_method <- init_method

    metadata_df <- data.frame(
        num_comb = py_main$num_comb,
        total_lr_time = py_main$total_lr_time,
        total_refit_time = py_main$total_refit_time,
        initial_opt_time = py_main$initial_opt_time,
        initial_reg_time = py_main$initial_reg_time,
        initial_hess_time = py_main$initial_hess_time,
        gmm_param_length = py_main$gmm_param_length,
        reg_param_length = py_main$reg_param_length,
        df=df,
        lo_num_times=lo_num_times,
        init_method=init_method)

    return(list(refit_err_df=refit_err_df,
                metadata_df=metadata_df))
}


LoadInitialFitIntoPyMain <- function(df) {
    py_run_string("
df = " %_% df %_% "
degree = 3
num_components = 18

load_filename = saving_gmm_utils.get_initial_fit_filename(
    df=df, degree=degree, num_components=num_components)

comb_params_pattern, comb_params, gmm, regs, metadata = \
    saving_gmm_utils.load_initial_optimum_fit_only(
        os.path.join('../../fits', load_filename))
    ")
}



MeltErrorColumns <- function(refit_err_df, extra_cols=c()) {
    # We need to melt three columns together, an operation that I
    # don't believe reshape2 supports.
    num_lo_cols <- sum(grepl("^time[0-9]+", names(refit_err_df)))
    other_cols <-
        c("test", "method", "comb", "rereg", "gene", "df", "lo_num_times",
          extra_cols)
    GetColumnDfCols <- function(col) {
        cols <- c(paste(c("err", "train_err", "time"), col, sep=""), other_cols)
        col_df <- refit_err_df[, cols]
        col_df <- rename(col_df, err=cols[1], train_err=cols[2], time=cols[3])
        return(col_df)
    }

    refit_err_melt <-
        do.call(dplyr::bind_rows, lapply(1:num_lo_cols, GetColumnDfCols)) %>%
        mutate(e_diff=err - train_err) %>%
        melt(id.vars=c(other_cols, "time"),
             value.name="value", variable.name="measure")

    return(refit_err_melt)
}
