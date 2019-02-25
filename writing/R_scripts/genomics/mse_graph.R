w <- 1.1
if (single_column) {
  grid_ncol <- 2
  grid_widths <- c(w, 2 - w) # For single-column paper
} else {
  grid_ncol <- 1
  grid_widths <- 1.0
}

mse_breaks <- c(
  "Exact CV", "IJ", "Training error", "Test error")
mse_graph_colors <- GetGraphColors(mse_breaks)
stopifnot(setequal(mse_breaks, unique(genomics_env$mse_summary_df$method_label)))

ggplot(genomics_env$mse_summary_df) +
geom_ribbon(aes(x=df, ymin=mean - 2 * se, ymax=mean + 2 * se,
    fill=method_label), alpha=0.2) +
geom_point(aes(x=df, y=mean, color=method_label)) +
geom_line(aes(x=df, y=mean, color=method_label)) +
scale_fill_manual(
  breaks=mse_breaks,
  values=mse_graph_colors,
  name=NULL) +
scale_color_manual(
  breaks=mse_breaks,
  values=mse_graph_colors,
  name=NULL) +
facet_grid(. ~ k_label, scales="free") +
xlab("Degrees of freedom") +
ylab("MSE on held out timepoints") +
theme(legend.title=element_blank()) +
theme(legend.position="top",
      legend.text=element_text(margin=margin(0, 10, 0, 0, "pt")))
