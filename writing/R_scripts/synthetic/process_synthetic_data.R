#########################################
# Develop and debug your graphs in R

library(ggplot2)
library(reshape2)
library(dplyr)

knitr_debug <- FALSE

setwd("/home/rgiordan/Documents/git_repos/AISTATS2019SwissArmyIJ/writing/R_scripts")
git_repo_loc <- system("git rev-parse --show-toplevel", intern=TRUE)
single_column <- FALSE # Just so initialize can load.
source(file.path(git_repo_loc, "writing/R_scripts/initialize.R"))

output_path <- file.path(data_path, "synthetic")

synth_results <- rbind(
  read.delim(
    file.path(data_path, "synthetic/synthetic_cv_SyntheticLogistic_N=1000_D=100.txt"),
    sep="", header=FALSE) %>%
    mutate(model="logistic"),
  read.delim(
    file.path(data_path, "synthetic/synthetic_cv_SyntheticPoisson_N=1000_D=100.txt"),
    sep="", header=FALSE) %>%
    mutate(model="poisson"),
  read.delim(
    file.path(data_path, "synthetic/synthetic_cv_SyntheticProbit_N=1000_D=100.txt"),
    sep="", header=FALSE) %>%
    mutate(model="probit")
)

# I have it on good faith that these are the column names.
names(synth_results)[1:4] <- c("train_error", "test_error", "exact_cv", "approx_cv")
stopifnot(unique(table(synth_results$ind)) == 3) # sanity check

synth_results_graph <-
  arrange(synth_results, test_error) %>%
  group_by(model) %>%
  mutate(ind=1:n()) %>%
  melt(id.vars=c("model", "ind"))

ggplot(synth_results_graph) +
  geom_line(aes(x=ind, y=value, color=variable)) +
  facet_grid(~ model, scales="free")

################

MakeMethodRow <- function(method, method_legend) {
  data.frame(method=method, method_legend=method_legend)
}
unique(synth_results_graph$variable)
synth_method_legend <- rbind(
  MakeMethodRow("train_error", "Training error"),
  MakeMethodRow("exact_cv", "Exact CV"),
  #MakeMethodRow("approx_cv", "Approximate CV")
  MakeMethodRow("approx_cv", "IJ")
)


MakeModelRow <- function(model, model_legend) {
  data.frame(model=model, model_legend=model_legend)
}
unique(synth_results_graph$model)
synth_model_legend <- rbind(
  MakeModelRow("probit", "Probit regression"),
  MakeModelRow("logistic", "Logistic regression"),
  MakeModelRow("poisson", "Poisson regression")
)

synth_results_diff <-
  inner_join(filter(synth_results_graph, variable  != "test_error"),
             filter(synth_results_graph, variable == "test_error"),
             by=c("ind", "model"), suffix=c("", "_test")) %>%
  mutate(diff = value - value_test) %>%
  dcast(model + ind ~ variable, value.var="diff") %>%
  arrange(train_error) %>%
  ungroup() %>% group_by(model) %>%
  mutate(ind=1:n()) %>%
  melt(id.vars=c("model", "ind")) %>%
  rename(method=variable) %>%
  filter(model != "probit") %>%
  inner_join(synth_method_legend, by="method") %>%
  inner_join(synth_model_legend, by="model")
head(synth_results_diff)

# keep synth_results_diff

ggplot(synth_results_diff) +
  geom_line(aes(x=ind, y=value, color=method_legend, linetype=method_legend), lwd=1) +
  geom_hline(aes(yintercept=0)) +
  xlab("Trial number, sorted by training set error difference") +
  ylab("Difference from test set error") +
  facet_grid(~ model_legend, scales="free") +
  guides(color=guide_legend(title=NULL)) +
  guides(linetype=guide_legend(title=NULL))



###############
# Timing plot

MakeTimingMethodRow <- function(variable, variable_legend) {
  data.frame(variable=variable, variable_legend=variable_legend)
}
timing_method_legend <- rbind(
  MakeTimingMethodRow("exact_runtime", "Exact CV"),
  # MakeTimingMethodRow("approx_runtime", "Approximate CV"),
  MakeTimingMethodRow("approx_runtime", "IJ"),
  MakeTimingMethodRow("linear_scaling", "Linear scaling\n(for reference)")
)


timings <- read.delim(
  file.path(data_path, "synthetic/synthetic_timings.txt"),
  sep="", header=FALSE)

# Educated guess
colnames(timings) <- c("data_size", "exact_runtime", "approx_runtime")
timings_graph <-
  melt(timings, id.vars="data_size")
ggplot(timings_graph) +
  geom_point(aes(x=data_size, y=value, color=variable)) +
  geom_line(aes(x=data_size, y=value, color=variable))

timings_graph_filter <- filter(timings_graph, data_size > 1000)
min_data_size <- min(timings_graph_filter$data)
target_intercept <-
  log10(min(filter(timings_graph_filter, data_size == min_data_size)$value)) - 1
straight_line_intercept <- target_intercept - log10(min(timings_graph_filter$data))

timings_graph_scaling <-
  dcast(timings_graph_filter, data_size ~ variable, value.var="value") %>%
  mutate(linear_scaling=10 ^(straight_line_intercept) * data_size) %>%
  melt(id.vars="data_size") %>%
  inner_join(timing_method_legend, by="variable")
#timings_graph_scaling

ggplot(timings_graph_scaling) +
  geom_line(aes(x=log10(data_size), y=log10(value), color=variable_legend, linetype=variable_legend), lwd=1) +
  guides(color=guide_legend(title=NULL)) +
  guides(linetype=guide_legend(title=NULL)) +
  xlab("Log10(data size)") + ylab("Log10(Runtime)")

summary(lm(log10(exact_runtime) ~ log10(data_size), filter(timings, data_size > 1000)))
summary(lm(log10(approx_runtime) ~ log10(data_size), filter(timings, data_size > 1000)))



###########################
# Save results

save(timings_graph_scaling, synth_results_diff,
     file=file.path(output_path, "synthetic_results.Rdata"))
