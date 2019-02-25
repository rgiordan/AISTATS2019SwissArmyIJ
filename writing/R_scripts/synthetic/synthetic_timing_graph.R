legend_breaks <- c(
    # "Exact CV", "Approximate CV", "Training error",
    "Exact CV", "IJ", "Training error",
    "Linear scaling\n(for reference)")
graph_colors <- GetGraphColors(legend_breaks)

# Make sure we typed the legends correctly.
stopifnot(setequal(
    unique(c(as.character(synthetic_env$synth_results_diff$method_legend),
             as.character(synthetic_env$timings_graph_scaling$variable_legend))),
                         legend_breaks))

graph_linetypes <- 1:4
names(graph_linetypes) <- legend_breaks

color_scale <- scale_color_manual(
  breaks=legend_breaks,
  values=graph_colors,
  name=NULL
)

linetype_scale <- scale_linetype_manual(
  breaks=legend_breaks,
  values=graph_linetypes,
  name=NULL
)

time_values <- unique(synthetic_env$timings_graph_scaling$value)
time_log10_range <-
  10 ^ seq(floor(log10(min(time_values))) - 1,
  ceiling(log10(max(time_values))) + 1)

data_sizes <- unique(synthetic_env$timings_graph_scaling$data_size)
data_size_log10_range <-
  c(min(data_sizes), median(data_sizes), max(data_sizes))

  if (single_column) {
      grid_ncol <- 2
      grid_widths <- c(1.5, 1) # For single-column paper
  } else {
      grid_ncol <- 1
      grid_widths <- 1.0
  }


ggplot(filter(synthetic_env$timings_graph_scaling, variable != "linear_scaling")) +
  geom_line(aes(x=data_size, y=value,
                color=variable_legend,
                linetype=variable_legend), lwd=1) +
  xlab("Data size N (log10 scale)") + ylab("Seconds (log10 scale)") +
  scale_x_log10(breaks=data_size_log10_range) +
  scale_y_log10(breaks=time_log10_range) +
  color_scale + linetype_scale +
  theme(legend.position="top",
        legend.text=element_text(margin=margin(0, 10, 0, 0, "pt")))
