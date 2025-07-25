
#LOAD DATA
library(lubridate)
library(dplyr)

case_study <- "feeagh" #feeagh or sau
dir <- paste0("~/Documents/intoDBP/driver_attribution_fdom/", case_study, "/")

rf <- read.csv(paste0(dir,"output/importance/imp_rf.csv"))
colnames(rf) <- c("feature", "importance")
xgb <- read.csv(paste0(dir,"output/importance/imp_xgb.csv"))
xgb <- xgb[c("Feature", "Gain")]
colnames(xgb) <- c("feature", "importance")
lgb <- read.csv(paste0(dir,"output/importance/imp_lgb.csv"))
lgb <- lgb[c("Feature", "Gain")]
colnames(lgb) <- c("feature", "importance")
cbt <- read.csv(paste0(dir,"output/importance/imp_cbt.csv"))
cbt <- cbt[c("var", "importance")]
colnames(cbt) <- c("feature", "importance")

#BUILD PLOT
if (case_study=="sau"){
  title_p <- "Sau Reservoir (ES)"
}else{
  title_p <- "Lough Feeagh (IE)"
}
# First, let's organize the data into a single data frame
library(ggplot2)
library(dplyr)
library(tidyr)

# Create data frames for each model
rf_data <- data.frame(
  feature = rf$feature,
  rf_importance = rf$importance,
  stringsAsFactors = FALSE
)

xgb_data <- data.frame(
  feature = xgb$feature,
  xgb_importance = xgb$importance,
  stringsAsFactors = FALSE
)

lgb_data <- data.frame(
  feature = lgb$feature,
  lgb_importance = lgb$importance,
  stringsAsFactors = FALSE
)

cbt_data <- data.frame(
  feature = cbt$feature,
  cbt_importance = cbt$importance,
  stringsAsFactors = FALSE
)

# Combine all data
all_data <- rf_data %>%
  full_join(xgb_data, by = "feature") %>%
  full_join(lgb_data, by = "feature") %>%
  full_join(cbt_data, by = "feature")

write.csv(all_data, paste0(dir,"output/importance/imp_all.csv"), 
                quote=F,row.names = F)

# Convert to long format for ggplot
long_data <- all_data %>%
  pivot_longer(cols = -feature, names_to = "model", values_to = "importance") %>%
  mutate(model = gsub("_importance", "", model))

# Order features by average importance across models
feature_order <- long_data %>%
  group_by(feature) %>%
  summarise(avg_importance = mean(importance)) %>%
  arrange((avg_importance)) %>%
  pull(feature)

long_data$feature <- factor(long_data$feature, levels = feature_order)

# Create a grouped bar plot
ggplot(long_data, aes(x = feature, y = importance*100, fill = model)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  scale_fill_manual(values = c("rf" = "#cce5ff", 
                               "xgb" = "#ffe599", 
                               "lgb" = "#b9e0a5", 
                               "cbt" = "#e6d0de")) +
  #scale_fill_manual(values = c("rf" = "#1f77b4", 
  #                             "xgb" = "#ff7f0e", 
  #                             "lgb" = "#2ca02c", 
  #                             "cbt" = "#d62728")) +
  labs(title = "Feature Importance Across Models",
       x = "Feature",
       y = "Importance",
       fill = "Model") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5))


psave <- ggplot(long_data, aes(x = importance*100, y = feature, fill = model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.7) +
  scale_fill_manual(values = c("rf" = "#cce5ff", 
                               "xgb" = "#ffe599", 
                               "lgb" = "#b9e0a5", 
                               "cbt" = "#e6d0de")) +
  #scale_fill_manual(values = c("rf" = "#1f77b4", 
  #                             "xgb" = "#ff7f0e", 
  #                             "lgb" = "#2ca02c", 
  #                             "cbt" = "#d62728"),
  #                  labels = c("Random Forest", "XGBoost", "LightGBM", "CatBoost")) +
  labs(title = title_p,
       x = "Importance Score (%)",
       y = "Driver",
       fill = "") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10),
        legend.position = "top",
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank()) +
  #scale_x_continuous(limits = c(0, 55),expand = expansion(mult = c(0, 0.05))) +  # Add some padding on right
  scale_x_continuous(expand = expansion(mult = c(0, 0.05))) +  # Add some padding on right
  coord_cartesian(clip = "off")  # Allow text to extend beyond plot area
psave

pdf(paste0(dir,"output/importance_score.pdf"))
print(psave)
dev.off()
