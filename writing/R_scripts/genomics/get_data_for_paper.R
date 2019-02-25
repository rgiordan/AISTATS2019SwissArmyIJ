# This script post-processes the output of
# AISTATS2019SwissArmyIJ/genomics/examine_and_save_results.ipynb,
# and produces the file genomics_data_for_paper.Rdata, which is used, per its name,
# for the graphs in the actual paper.
#
# We assume that the Rdata file produced by examine_and_save_results.ipynb is linked to
# in ``genomics_data_path``.  This Rdata file should contain the following objects:
#
# refit_err_summary
# metadata_df
# diff_corr
# err_corr

library(tidyverse)

setwd("/home/rgiordan/Documents/git_repos/AISTATS2019SwissArmyIJ/writing/R_scripts")
git_repo_loc <- system("git rev-parse --show-toplevel", intern=TRUE)
knitr_debug <- FALSE
single_column <- TRUE
source(file.path(git_repo_loc, "writing/R_scripts/initialize.R"))
genomics_data_path <- file.path(data_path, "genomics/")

load(file.path(genomics_data_path, "paper_results_init_kmeans_rereg_FALSE.Rdata"))


####################################################################
# Make the plot comparing the errors for different k and df.

head(refit_err_summary)

MakeKLabel <- function(k) { return(data.frame(lo_num_times=k, k_label=sprintf("K = %d", k))) }
k_label_df <- do.call(rbind, lapply(unique(refit_err_summary$lo_num_times), MakeKLabel))

MakeDFLabel <- function(df) { return(data.frame(df=df, df_label=sprintf("df = %d", df)))}
df_label_df <- do.call(rbind, lapply(unique(refit_err_summary$df), MakeDFLabel))

MakeMethodLabel <- function(method, method_label) {
  return(data.frame(output=method, method_label=method_label))
}
unique(refit_err_summary$output)
method_label_df <- rbind(
  MakeMethodLabel("cv_in_sample", "Exact CV"),
  #MakeMethodLabel("lin_in_sample", "Approximate CV"),
  MakeMethodLabel("lin_in_sample", "IJ"),
  MakeMethodLabel("test_error", "Test error"),
  MakeMethodLabel("train_error", "Training error")
)

mse_summary_df <-
  refit_err_summary %>%
  inner_join(method_label_df, by="output") %>%
  inner_join(k_label_df, by="lo_num_times") %>%
  inner_join(df_label_df, by="df")

head(mse_summary_df)

ggplot(mse_summary_df) +
  geom_ribbon(aes(x=df, ymin=mean - 2 * se, ymax=mean + 2 * se, fill=method_label), alpha=0.2) +
  geom_point(aes(x=df, y=mean, color=method_label)) +
  geom_line(aes(x=df, y=mean, color=method_label)) +
  facet_grid(. ~ k_label, scales="free") +
  xlab("Degrees of freedom") +
  ylab("Mean squared error on held out timepoints") +
  theme(legend.title=element_blank())


#######################
# Timing results

# approx_cv_label <- "Approximate CV"
approx_cv_label <- "IJ"
exact_cv_label <- "Exact CV"
MakeTaskLabel <- function(task, task_label) { data.frame(task=task, task_label=task_label) }
task_labels <- rbind(
  MakeTaskLabel("initial_opt_time", "Initial fit"),
  MakeTaskLabel("total_lr_time", "Matrix multiplication"),
  MakeTaskLabel("total_ij_time", approx_cv_label),
  MakeTaskLabel("lr_hess_time", "Calculating and inverting Hessian"),
  MakeTaskLabel("total_refit_time", exact_cv_label)
)

timing_graph_df1 <-
  metadata_df %>%
  select(df, lo_num_times, total_refit_time, lr_hess_time, initial_opt_time, total_lr_time) %>%
  mutate(total_ij_time=lr_hess_time + total_lr_time) %>%
  select(-lr_hess_time, -total_lr_time) %>%
  melt(id.vars=c("lo_num_times", "df"), variable.name="task") %>%
  inner_join(task_labels, by="task") %>%
  inner_join(k_label_df, by="lo_num_times") %>%
  inner_join(df_label_df, by="df")

head(timing_graph_df1)

# Because the initial fit contributes both to CV and the IJ, append
# the rows to both the meta tasks.
timing_graph_df <-
  filter(timing_graph_df1, task %in% c("total_refit_time", "total_ij_time")) %>%
  mutate(meta_task=task_label) %>%
  bind_rows(
    filter(timing_graph_df1, task %in% c("initial_opt_time")) %>%
      mutate(meta_task=exact_cv_label)) %>%
  bind_rows(
    filter(timing_graph_df1, task %in% c("initial_opt_time")) %>%
      mutate(meta_task=approx_cv_label))

ggplot(timing_graph_df) +
  geom_bar(aes(fill=task_label, y=value, x=df, group=meta_task),
           stat="identity", position=position_dodge()) +
  ylab("Seconds") +
  ggtitle("Timing comparison") +
  theme(axis.text.x = element_text(angle = 0, hjust = 1)) +
  theme(legend.title=element_blank()) +
  theme(axis.title.x=element_blank()) +
  facet_wrap(~ paste("df = 4,...,8", k_label, sep="\t"),
             strip.position = "bottom", scales = "free_x") +
  theme(panel.spacing = unit(0, "lines"),
        strip.background = element_blank(),
        strip.placement = "outside") + 
  theme(legend.position="top",
        legend.text=element_text(margin=margin(0, 10, 0, 0, "pt")))

#######################
# Save for the paper

save(timing_graph_df, mse_summary_df, metadata_df,
     file=file.path(genomics_data_path, "genomics_data_for_paper.Rdata"))
