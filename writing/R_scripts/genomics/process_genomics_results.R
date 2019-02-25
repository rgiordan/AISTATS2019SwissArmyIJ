##########################
# Some plots

library(tidyverse)

setwd("/home/rgiordan/Documents/git_repos/AISTATS2019SwissArmyIJ/writing/R_scripts")
git_repo_loc <- system("git rev-parse --show-toplevel", intern=TRUE)
knitr_debug <- FALSE
source(file.path(git_repo_loc, "writing/R_scripts/initialize.R"))
genomics_data_path <- file.path(data_path, "genomics/MSE_results/")

# suffix can be jack, lr, or test
LoadMSEDataFrame <- function(df, k, suffix){
  # the degrees of freedom, the number left out(k)
  # and whether you want to true jackknife mses 
  # or the linear response
  
  file_suffix <- sprintf("_%s_mses.csv", suffix)

  file <- paste('df', df, '_leaveout_k', k, file_suffix, sep = '')
  mses_mat <- read.csv(file.path(genomics_data_path, file), sep = ' ', row.names = 1)
  mses_df <- data.frame(mses_mat)
  mses_df$trial <- rownames(mses_mat)
  mses_melt_df <- melt(mses_df, id.vars="trial")
  mses_melt_df$k <- k
  mses_melt_df$df <- df
  mses_melt_df$method <- suffix
  return(mses_melt_df)
}

# sanity check
lr_mses <- LoadMSEDataFrame(7, 1, suffix="jack")
head(lr_mses)

result_lists <- list()
for (df in 4:8) { for (k in c(1, 2, 3)) { for (suffix in c("jack", "lr", "test")) {
  cat(".")
  # if (suffix == "test" && k == 8) {
  #   print("Missing this one")
  # } else {
  result_lists[[length(result_lists) + 1]] <- LoadMSEDataFrame(df=df, k=k, suffix)
  # }
}}}
print("Done.")
result_df <- do.call(rbind, result_lists)


# Remove some bulky text
result_df$trial_num <- as.numeric(sub("^trial", "", result_df$trial))
result_df$datapoint <- as.numeric(sub("^datapoint", "", result_df$variable))
mse_result_df <- select(result_df, -variable, -trial)


######################
# Save for further processing

save(mse_result_df, file=file.path(genomics_data_path, "mse_results.Rdata"))




