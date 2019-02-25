w <- 1.1
if (single_column) {
  grid_ncol <- 2
  grid_widths <- c(w, 2 - w) # For single-column paper
} else {
  grid_ncol <- 1
  grid_widths <- 1.0
}

timing_breaks <- c(
  "Exact CV", "IJ", "Initial fit")
timing_graph_colors <- GetGraphColors(timing_breaks)
stopifnot(setequal(timing_breaks, unique(genomics_env$timing_graph_df$task_label)))

ggplot(genomics_env$timing_graph_df) +
  geom_bar(aes(fill=task_label, y=value, x=df, group=meta_task),
           stat="identity", position=position_dodge()) +
  ylab("Seconds") +
  xlab("") +
  #ggtitle("Timing comparison") +
  scale_fill_manual(
    breaks=timing_breaks,
    values=timing_graph_colors,
    name=NULL) +
  theme(axis.text.x = element_text(angle = 0, hjust = 1)) +
  theme(legend.title=element_blank()) +
  theme(axis.title.x=element_blank()) +
  facet_wrap(~ paste(k_label, "df = 4,...,8", sep=",  "),
             strip.position = "bottom", scales = "free_x") +
  theme(panel.spacing = unit(0, "lines"),
        strip.background = element_blank(),
        strip.placement = "outside") +
   theme(legend.position="top",
         legend.text=element_text(margin=margin(0, 10, 0, 0, "pt")))
