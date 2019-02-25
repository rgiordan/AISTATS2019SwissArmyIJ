# Load the data you need into an environment here.
synthetic_env <- LoadIntoEnvironment(file.path(data_path, "synthetic/synthetic_results.Rdata"))

genomics_data_path <- file.path(data_path, "genomics/")
genomics_env <- LoadIntoEnvironment(file.path(genomics_data_path, "genomics_data_for_paper.Rdata"))
