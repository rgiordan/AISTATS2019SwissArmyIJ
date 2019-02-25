# Use this script to debug and edit the knit graphs without re-compiling in latex.

# This must be run from within the AISTATS2019SwissArmyIJ git repo.
git_repo_loc <- system("git rev-parse --show-toplevel", intern=TRUE)

knitr_debug <- FALSE # Set to true to see error output
simple_cache <- FALSE # Set to true to cache knitr output for this analysis.
source(file.path(git_repo_loc, "writing/R_scripts/initialize.R"))
source(file.path(git_repo_loc, "writing/R_scripts/load_data.R"))

source(file.path(paper_directory, "R_scripts/synthetic/synthetic_graphs.R"))
