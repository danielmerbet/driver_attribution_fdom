library(ggplot2)
library(moments)   # skewness
library(dplyr)

case_study <- "sau" #feeagh or sau
dir <- paste0("~/Documents/intoDBP/driver_attribution_fdom/",case_study, "/")
#load drivers (meteorology, soil,  streamflow and all possible variables)
data <- read.csv(paste0(dir, "data/data.csv"))

df_plot <- data.frame(x = data$fdom)

#initial histogram and density plot
ggplot(df_plot, aes(x)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30,
                 fill = "grey80",
                 color = "black") +
  geom_density(size = 1) +
  labs(
    x = "fDOM",
    y = "Density",
    title = "Histogram and Kernel Density Estimation"
  ) +
  theme_minimal()

#summary
x <- (data$fdom)
summary_stats <- tibble(
  mean      = mean(x, na.rm = TRUE),
  median    = median(x, na.rm = TRUE),
  sd        = sd(x, na.rm = TRUE),
  min       = min(x, na.rm = TRUE),
  max       = max(x, na.rm = TRUE),
  skewness  = skewness(x, na.rm = TRUE)
)

print(summary_stats)

#add outlier analysis to histogram
Q1  <- quantile(x, 0.25, na.rm = TRUE)
Q3  <- quantile(x, 0.75, na.rm = TRUE)
IQR <- Q3 - Q1

lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

outliers <- x[x < lower_bound | x > upper_bound]

cat("Number of outliers:", length(outliers), "\n")

ggplot(df_plot, aes(x)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30,
                 fill = "grey80",
                 color = "black") +
  geom_density(size = 1) +
  geom_vline(xintercept = c(lower_bound, upper_bound),
             linetype = "dashed",
             color = "red") +
  geom_text(x = max(x)-10,
            y = 0.05,
            label = paste0(
              "Mean: ", round(summary_stats[1],2), "\n",
              "Median: ", round(summary_stats[2],2), "\n",
              "Stand.Dev: ", round(summary_stats[3],2), "\n",
              "Min: ", round(summary_stats[4],2), "\n",
              "Max: ", round(summary_stats[5],2), "\n",
              "Skewness: ", round(summary_stats[6],2), ""
            ))+
  labs(
    title = "Histogram, KDE, and Outlier Thresholds",
    x = "Value",
    y = "Density"
  ) +
  theme_minimal()
