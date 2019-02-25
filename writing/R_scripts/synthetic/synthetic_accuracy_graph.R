
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

if (single_column) {
  grid_ncol <- 2
  grid_widths <- c(1.5, 1) # For single-column paper
} else {
  grid_ncol <- 1
  grid_widths <- 1.0
}


ggplot(synthetic_env$synth_results_diff) +
  geom_line(aes(x=ind, y=value, color=method_legend, linetype=method_legend), lwd=1) +
  geom_hline(aes(yintercept=0)) +
  xlab("Trial number, sorted by\ntraining set error difference") +
  ylab("Difference from test set error") +
  facet_grid(~ model_legend, scales="free") +
  color_scale + linetype_scale +
  theme(legend.position="top",
        legend.text=element_text(margin=margin(0, 10, 0, 0, "pt")))
