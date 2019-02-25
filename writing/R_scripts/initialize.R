# Initialize R for knitr.

library(knitr)
library(dplyr)
library(reshape2)
library(ggplot2)
library(xtable)
library(gridExtra)
library(latex2exp)

# This must be run from within the git repo, obviously.
git_repo_loc <- system("git rev-parse --show-toplevel", intern=TRUE)

paper_directory <- file.path(git_repo_loc, "writing/")
data_path <- file.path(paper_directory, "data/")

# opts_chunk$set(fig.width=4.9, fig.height=3)
opts_chunk$set(fig.pos='!h', fig.align='center', dev='png', dpi=300)
opts_chunk$set(echo=knitr_debug, message=knitr_debug, warning=knitr_debug)

# Set the default ggplot theme
theme_set(theme_bw())

# Load into an environment rather than the global space
LoadIntoEnvironment <- function(filename) {
  my_env <- environment()
  load(filename, envir=my_env)
  return(my_env)
}

# Define LaTeX macros that will let us automatically refer
# to simulation and model parameters.
DefineMacro <- function(macro_name, value, digits=3) {
  #sprintf_code <- paste("%0.", digits, "f", sep="")
  value_string <- format(value, big.mark=",", digits=digits, scientific=FALSE)
  cat("\\newcommand{\\", macro_name, "}{", value_string, "}\n", sep="")
}

# aspect ratio refers to height / width.
if (single_column) {
  # This is for the arxiv (single-column) version.
  base_aspect_ratio <- 3.5 / (5 * 2)
  base_image_width <- 4.9 * 2
} else {
  # This is for the AISTATS (two-column) submission.
  base_aspect_ratio <- 8 / (5 * 2)
  base_image_width <- 4.
}

SetImageSize <- function(aspect_ratio, image_width=base_image_width) {
  ow <- "0.98\\linewidth"
  oh <- sprintf("%0.3f\\linewidth", aspect_ratio * 0.98)
  fw <- image_width
  fh <- image_width * aspect_ratio
  opts_chunk$set(out.width=ow,
                 out.height=oh,
                 fig.width=fw,
                 fig.height=fh)
}


SetFullImageSize <- function() SetImageSize(
    aspect_ratio=base_aspect_ratio, image_width=base_image_width)

# Default to a full image.
SetFullImageSize()

# A convenient funciton for extracting only the legend from a ggplot.
# Taken from
# http://www.sthda.com/english/wiki/ggplot2-easy-way-to-mix-multiple-graphs-on-the-same-page-r-software-and-data-visualization
GetLegend <- function(myggplot){
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}


# Define common colors.
GGColorHue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

GetGraphColors <- function(legend_breaks) {
  stopifnot(length(legend_breaks) <= 4)
  graph_colors <- GGColorHue(4)[1:length(legend_breaks)]
  names(graph_colors) <- legend_breaks
  return(graph_colors)
}
