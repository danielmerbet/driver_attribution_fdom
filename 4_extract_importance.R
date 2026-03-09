library(lubridate)
library(dplyr)

library(lubridate)
library(dplyr)

case_study <- "sau" #select one: feeagh or sau
model_name_toplot <- "CTB" #select one: c("RF", "XGB", "LGB", "CTB", "LM", "KNN", "SVR", "Blended)

#if run in RStudio use this command:
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

dir <- paste0(getwd(),"/",case_study, "/")

#Tuned hyperparameters were obtained in the previous code (2_hyperparameter_tuning.R)
#the best combination of hyperparameters were used here

# Load data
data <- read.csv(paste0(dir, "data/data.csv"))
data$date <- as.Date(data$date)

# Add cos of julian day
data$cyday <- cos(yday(data$date) * pi / 180)

# Select columns
#if (case_study=="feeagh"){
#  data <- data[, c("swt", "sr", "st100", "st255", "sm100", "sm255", "doc_gwlf", "cyday", "fdom", "date")]
#}
#if (case_study=="sau"){
#  data <- data[, c("v", "st255", "sm100", "sm255", "cyday", "fdom", "date")] # "doc_gwlf",
#}

#load hyperparameters
load(paste0(dir, "output/hyperparameters/tuned_models.rdata"))

# Train/test split (last 15%)
n <- nrow(data)
n_holdout <- round(n * 0.15)

train_data <- data[1:(n - n_holdout), ]
test_data <- data[(n - n_holdout + 1):n, ]

# Remove date
X_train <- train_data %>% select(-fdom, -date)
y_train <- train_data$fdom

X_test <- test_data %>% select(-fdom, -date)
y_test <- test_data$fdom

#METRICS
# R2
r2 <- function(obs, pred) {
  cor(obs, pred)^2
}

# RMSE
rmse <- function(obs, pred) {
  sqrt(mean((obs - pred)^2))
}

# NSE
nse <- function(obs, pred) {
  numerator <- sum((obs - pred)^2)
  denominator <- sum((obs - mean(obs))^2)
  1 - (numerator / denominator)
}

# KGE
kge <- function(obs, pred) {
  r <- cor(pred, obs, use = "pairwise.complete.obs")
  alpha <- sd(pred, na.rm = TRUE) / sd(obs, na.rm = TRUE)
  beta <- mean(pred, na.rm = TRUE) / mean(obs, na.rm = TRUE)
  1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
}

preds_training <- c()
#MACHINE LEARNING METHODS
library(randomForest)
set.seed(123)
rf <- randomForest(fdom ~ . -date, data = train_data, ntree = 1000, 
                   mtry=tuned_models$rf$bestTune$mtry)
pred_rf <- predict(rf, newdata = test_data)
preds_training <- cbind(preds_training,
                        predict(rf, newdata = train_data))
write.csv(importance(rf)/sum(importance(rf)), 
          file=paste0(dir,"output/importance/imp_rf.csv"),
          quote = F)

library(xgboost)
set.seed(123)
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test))

params <- list(objective = "reg:squarederror", 
               max_depth = tuned_models$xgb$bestTune$max_depth,
               eta=tuned_models$xgb$bestTune$eta, 
               gamma=tuned_models$xgb$bestTune$gamma,
               colsample_bytree= tuned_models$xgb$bestTune$colsample_bytree,
               min_child_weight=tuned_models$xgb$bestTune$min_child_weight,
               subsample=tuned_models$xgb$bestTune$subsample)
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = tuned_models$xgb$bestTune$nrounds)
pred_xgb <- predict(xgb_model, newdata = dtest)
xgb.importance(model = xgb_model)
preds_training <- cbind(preds_training,
                        predict(xgb_model, newdata = dtrain))
write.csv(xgb.importance(model = xgb_model), 
          file=paste0(dir,"output/importance/imp_xgb.csv"),
          quote = F)

library(lightgbm)
set.seed(123)
lgb_train <- lgb.Dataset(data = as.matrix(X_train), label = y_train)
params <- list(
  objective = tuned_models$lgb$objective,
  metric = tuned_models$lgb$metric,
  learning_rate = tuned_models$lgb$learning_rate,
  num_leaves = tuned_models$lgb$num_leaves,
  max_depth = tuned_models$lgb$max_depth,
  feature_fraction = tuned_models$lgb$feature_fraction,
  bagging_fraction = tuned_models$lgb$bagging_fraction
)
lgb_model <- lgb.train(params = params, 
                       data = lgb_train, nrounds = 2000)
pred_lgb <- predict(lgb_model, newdata = as.matrix(X_test))
preds_training <- cbind(preds_training, predict(lgb_model, newdata = as.matrix(X_train)))


lgb.importance(lgb_model)
write.csv(lgb.importance(lgb_model), 
          file=paste0(dir,"output/importance/imp_lgb.csv"),
          quote = F)

library(catboost)
set.seed(123)
train_pool <- catboost.load_pool(data = as.matrix(X_train), label = y_train)
test_pool <- catboost.load_pool(data = as.matrix(X_test))

params <- list(
  loss_function = tuned_models$ctb$loss_function, 
  iterations = tuned_models$ctb$iterations,
  depth = tuned_models$ctb$depth,
  learning_rate = tuned_models$ctb$learning_rate,
  l2_leaf_reg = tuned_models$ctb$l2_leaf_reg,
  logging_level = tuned_models$ctb$logging_level
)
cat_model <- catboost.train(learn_pool = train_pool, 
                            params = params)
pred_ctb <- catboost.predict(cat_model, pool = test_pool)
preds_training <- cbind(preds_training,
                        catboost.predict(cat_model, pool = train_pool))
data.frame(var=colnames(X_train),
           importance=catboost.get_feature_importance(cat_model)/sum(catboost.get_feature_importance(cat_model)))

write.csv(data.frame(var=colnames(X_train),
                     importance=catboost.get_feature_importance(cat_model)/sum(catboost.get_feature_importance(cat_model))), 
          file=paste0(dir,"output/importance/imp_cbt.csv"),
          quote = F)


# Extract importance
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
