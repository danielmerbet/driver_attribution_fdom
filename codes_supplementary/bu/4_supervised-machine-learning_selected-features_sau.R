library(lubridate)
library(dplyr)
library(here)

setwd(here())


case_study <- "sau" #feeagh or sau
dir <- paste0("../",case_study, "/")

# Load data
data <- read.csv(paste0(dir, "data/data.csv"))
data$date <- as.Date(data$date)

# Add cos of julian day
data$cyday <- cos(yday(data$date) * pi / 180)

# Select columns
data <- data[, c("v", "st255", "sm100", "sm255", "cyday", "fdom", "date")] # "doc_gwlf",

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
rf <- randomForest(fdom ~ . -date, data = train_data, ntree = 300, mtry=2)
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

params <- list(objective = "reg:squarederror", max_depth = 6, eta=0.1, gamma=0,
               colsample_bytree=1, min_child_weight=1, subsample=1)
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 300)
pred_xgb <- predict(xgb_model, newdata = dtest)
xgb.importance(model = xgb_model)
preds_training <- cbind(preds_training,
                        predict(xgb_model, newdata = dtrain))
write.csv(xgb.importance(model = xgb_model), 
          file=paste0(dir,"output/importance/imp_xgb.csv"),
          quote = F)

# Save XGBoost model
xgb.save(xgb_model, paste0(dir, "output/models/xgb_model.model"))

# Export data for SHAP analysis in Python
write.csv(X_test, paste0(dir, "output/shap/X_test.csv"), row.names = FALSE)
write.csv(X_train, paste0(dir, "output/shap/X_train.csv"), row.names = FALSE)
write.csv(y_test, paste0(dir, "output/shap/y_test.csv"), row.names = FALSE)
write.csv(y_train, paste0(dir, "output/shap/y_train.csv"), row.names = FALSE)


library(lightgbm)
set.seed(123)
lgb_train <- lgb.Dataset(data = as.matrix(X_train), label = y_train)
params <- list(
  objective = "regression",
  metric = "rmse",
  learning_rate = 0.01,
  num_leaves = 31,
  max_depth = -1,
  feature_fraction = 1,
  bagging_fraction = 0.8
)
lgb_model <- lgb.train(params = params, 
                       data = lgb_train, nrounds = 500)
pred_lgb <- predict(lgb_model, newdata = as.matrix(X_test))
preds_training <- cbind(preds_training, predict(lgb_model, newdata = as.matrix(X_train)))


lgb.importance(lgb_model)
write.csv(lgb.importance(lgb_model), 
          file=paste0(dir,"output/importance/imp_lgb.csv"),
          quote = F)

# Save LightGBM model to file
lgb.save(lgb_model, paste0(dir, "output/models/lgb_model.txt"))


library(catboost)
set.seed(123)
train_pool <- catboost.load_pool(data = as.matrix(X_train), label = y_train)
test_pool <- catboost.load_pool(data = as.matrix(X_test))

params <- list(
  loss_function = "RMSE",
  iterations = 500,
  depth = 4,
  learning_rate = 0.05,
  l2_leaf_reg = 1,
  logging_level = "Silent"
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

# Save CatBoost model in binary format
catboost.save_model(cat_model,
                    model_path = paste0(dir, "output/models/cat_model.cbm"),
                    file_format = "cbm")

#liner regresion
set.seed(123)
lm_model <- lm(fdom ~ . -date, data = train_data)
pred_lm <- predict(lm_model, newdata = test_data)

preds_training <- cbind(preds_training,predict(lm_model, newdata = train_data))
#doesn't directly provide feature importance scores

library(kknn)
set.seed(123)
knn_model <- kknn::train.kknn(fdom ~ . -date, data = train_data, ks = 7) #
pred_knn <- predict(knn_model, newdata = test_data)
preds_training <- cbind(preds_training, predict(knn_model, newdata = train_data))
#doesn't directly provide feature importance scores

library(kernlab)
set.seed(123)
svr_model <- ksvm(fdom ~ . -date, data = train_data, kernel = "rbfdot", C = 10, sigma=0.1)
pred_svr <- predict(svr_model, newdata = test_data)
preds_training <- cbind(preds_training, predict(svr_model, newdata = train_data))
#doesn't directly provide feature importance scores


preds_testing <- cbind(pred_rf, pred_xgb, pred_lgb, pred_ctb, pred_lm, pred_knn, pred_svr)
colnames(preds_training) <- c("RF", "XGB", "LGB", "CTB", "LM", "KNN", "SVR")
colnames(preds_testing) <- c("RF", "XGB", "LGB", "CTB", "LM", "KNN", "SVR")
preds_training <- data.frame(preds_training)

library(caret)
library(Metrics)
set.seed(123)
# Train models separately, then combine
results <- list()

evaluate <- function(name, pred, obs = y_test) {
  results <<- append(results, list(list(
    Model = name,
    R2 = r2(obs, pred),
    RMSE = rmse(obs, pred),
    NSE = nse(obs, pred),
    KGE = kge(obs, pred)
  )))
}

evaluate("RF", pred_rf)
evaluate("XGB", pred_xgb)
evaluate("LGB", pred_lgb)
evaluate("CatBoost", pred_ctb)
evaluate("Linear", pred_lm)
evaluate("KNN", pred_knn)
evaluate("SVR", pred_svr)

results_df <- do.call(rbind.data.frame, results)
print(results_df)

#now, you should go to optmised_blending.R, in case you want to re-run de optimisation
w1 <- 0.586 #linear
w2 <- 0.414 #catboost
blended<- w1 * pred_lm + w2 * pred_ctb
blended_training <- w1 * preds_training$LM + w2 * preds_training$CTB

evaluate("Blended", blended)

results_df <- do.call(rbind.data.frame, results)
results_df <-data.frame(Model=results_df$Model, round(results_df[,2:5],2))
print(results_df)

write.csv(results_df, file=paste0(dir,"output/metrics_models_testing.csv"),
          quote = F, row.names = F)

#save for training:
results <- list()

evaluate <- function(name, pred, obs = y_train) {
  results <<- append(results, list(list(
    Model = name,
    R2 = r2(obs, pred),
    RMSE = rmse(obs, pred),
    NSE = nse(obs, pred),
    KGE = kge(obs, pred)
  )))
}

evaluate("RF", preds_training$RF)
evaluate("XGB", preds_training$XGB)
evaluate("LGB", preds_training$LGB)
evaluate("CatBoost", preds_training$CTB)
evaluate("Linear", preds_training$LM)
evaluate("KNN", preds_training$KNN)
evaluate("SVR", preds_training$SVR)

#now, you should go to optmised_blending.R, in case you want to re-run de optimisation
w1 <- 0.586 #linear
w2 <- 0.414 #catboost
blended<- w1 * pred_lm + w2 * pred_ctb
blended_training <- w1 * preds_training$LM + w2 * preds_training$CTB


evaluate("Blended", blended_training)

results_df <- do.call(rbind.data.frame, results)
results_df <-data.frame(Model=results_df$Model, round(results_df[,2:5],2))
print(results_df)

write.csv(results_df, file=paste0(dir,"output/metrics_models_training.csv"),
          quote = F, row.names = F)

#PLOT ALL MODELS

# Store training and test predictions
#meta_model_training <- lm(y_train ~ preds_training)

train_predictions_list <- list(
  RF = preds_training$RF,
  XGB = preds_training$XGB,
  LGB = preds_training$LGB,
  CatBoost = preds_training$CTB,
  Linear = preds_training$LM,
  KNN = preds_training$KNN,
  SVR = preds_training$SVR,
  Blended = blended_training
)

test_predictions_list <- list(
  RF = pred_rf,
  XGB = pred_xgb,
  LGB = pred_lgb,
  CatBoost = pred_ctb,
  Linear = pred_lm,
  KNN = pred_knn,
  SVR = pred_svr,
  Blended = blended
)

# Compute metrics for training and testing
train_metrics_list <- list()
test_metrics_list <- list()

for (model_name in names(train_predictions_list)) {
  train_metrics_list[[model_name]] <- list(
    R2 = r2(y_train, train_predictions_list[[model_name]]),
    RMSE = rmse(y_train, train_predictions_list[[model_name]]),
    NSE = nse(y_train, train_predictions_list[[model_name]]),
    KGE = kge(y_train, train_predictions_list[[model_name]])
  )
  
  test_metrics_list[[model_name]] <- list(
    R2 = r2(y_test, test_predictions_list[[model_name]]),
    RMSE = rmse(y_test, test_predictions_list[[model_name]]),
    NSE = nse(y_test, test_predictions_list[[model_name]]),
    KGE = kge(y_test, test_predictions_list[[model_name]])
  )
}

dates_train = train_data$date
y_train_actual = y_train
train_predictions_list = train_predictions_list

dates_test = test_data$date
y_test_actual = y_test
test_predictions_list = test_predictions_list

train_metrics_list = train_metrics_list
test_metrics_list = test_metrics_list

#title = "Supervised Machine Learning Approaches (Train + Test)",
title = ""
filename = paste0(dir, "output/model_comparison_train_test_selected.pdf")

# Combine predictions into a data frame
model_name_toplot <- "CatBoost"
all_train <- lapply(names(train_predictions_list), function(model_name_toplot) {
  data.frame(
    Date = dates_train,
    Value = train_predictions_list[[model_name_toplot]],
    Model = model_name_toplot,
    Period = "Train"
  )
})
df_train_pred <- do.call(rbind, all_train)
df_train_actual <- data.frame(Date = dates_train, Value = y_train_actual, Model = "Actual", Period = "Train")

all_test <- lapply(names(test_predictions_list), function(model_name_toplot) {
  data.frame(
    Date = dates_test,
    Value = test_predictions_list[[model_name_toplot]],
    Model = model_name_toplot,
    Period = "Test"
  )
})
df_test_pred <- do.call(rbind, all_test)
df_test_actual <- data.frame(Date = dates_test, Value = y_test_actual, Model = "Actual", Period = "Test")

# Combine everything
df_plot <- rbind(
  df_train_pred, df_train_actual,
  df_test_pred, df_test_actual
)

# Set default colors if not provided
#if (is.null(model_colors)) {
model_colors <- setNames(
  #c("black", rainbow(length(train_predictions_list))),
  c(rep("grey",length(train_predictions_list)),"black"),
  c( names(train_predictions_list),"zActual")
)
#}

obs_df_plot <- df_plot[df_plot$Model=="Actual",]

# Format metrics for display
#train
metric_text <- unlist(lapply(names(test_metrics_list), function(model_name) {
  train_metrics <- train_metrics_list[[model_name]]
  test_metrics <- test_metrics_list[[model_name]]
  paste0(model_name, ":  \nR2=", sprintf("%.2f", train_metrics$R2),
         " \nKGE=", sprintf("%.2f", train_metrics$KGE),
         " \nRMSE=", sprintf("%.2f", train_metrics$RMSE)
         #" \nNSE=", sprintf("%.2f", train_metrics$NSE)
         )
}))
#metric_text <-c("Training: ", metric_text)
metric_text<-c("Training: ", metric_text[grepl(model_name_toplot, metric_text)])

#test
metric_text_test <- unlist(lapply(names(test_metrics_list), function(model_name) {
  train_metrics <- train_metrics_list[[model_name]]
  test_metrics <- test_metrics_list[[model_name]]
  paste0(model_name, ": \nR2=", sprintf("%.2f", test_metrics$R2),
         " \nKGE=", sprintf("%.2f", test_metrics$KGE),
         " \nRMSE=", sprintf("%.2f", test_metrics$RMSE)
         #" \nNSE=", sprintf("%.2f", test_metrics$NSE)
         )
}))
#metric_text_test <-c("Testing: ", metric_text_test)
metric_text_test <-c("Testing: ", metric_text_test[grepl(model_name_toplot,metric_text_test)])

metric_text_str <- paste(metric_text, collapse = "\n")
metric_text_str_test <- paste(metric_text_test, collapse = "\n")

# Plot
#p <- ggplot(df_plot, aes(x = Date, y = Value, color = Model, linetype = Period)) +
#p <- ggplot(df_plot, aes(x = Date, y = Value, color = Model)) +
yi<-0;ye<-60

pdf(paste0(dir, "output/simulation1_",model_name_toplot,".pdf"), width = 8, height = 6)
plot(obs_df_plot$Date, obs_df_plot$Value, ylab="fDOM (QSU)", 
     xlab = "", xaxt="n", bty="n", ylim=c(yi,ye))
axis.Date(1, at = seq(as.Date(paste0(year(min(obs_df_plot$Date)),"-", "01-01")), 
                      as.Date(paste0(year(max(obs_df_plot$Date)),"-", "12-31")),
                      by = "6 months"), format = "%m-%Y")
#points(df_train_pred$Date[df_train_pred$Model=="RF"], 
#       df_train_pred$Value[df_train_pred$Model=="RF"], col="steelblue", pch = 19, cex=0.5)
#points(df_test_pred$Date[df_test_pred$Model=="RF"], 
#       df_test_pred$Value[df_test_pred$Model=="RF"], col="brown", pch = 19, cex=0.5)
#points(df_train_pred$Date[df_train_pred$Model=="XGB"], 
#       df_train_pred$Value[df_train_pred$Model=="XGB"], col="steelblue", pch = 19, cex=0.5)
#points(df_test_pred$Date[df_test_pred$Model=="XGB"], 
#       df_test_pred$Value[df_test_pred$Model=="XGB"], col="brown", pch = 19, cex=0.5)

#points(df_train_pred$Date[df_train_pred$Model=="LGB"], 
#       df_train_pred$Value[df_train_pred$Model=="LGB"], col="steelblue", pch = 19, cex=0.5)
#points(df_test_pred$Date[df_test_pred$Model=="LGB"], 
#       df_test_pred$Value[df_test_pred$Model=="LGB"], col="brown", pch = 19, cex=0.5)

if(model_name_toplot=="CatBoost"){
  points(df_train_pred$Date[df_train_pred$Model=="CatBoost"], 
         df_train_pred$Value[df_train_pred$Model=="CatBoost"], col="steelblue", pch = 19, cex=0.5)
  points(df_test_pred$Date[df_test_pred$Model=="CatBoost"], 
         df_test_pred$Value[df_test_pred$Model=="CatBoost"], col="magenta4", pch = 19, cex=0.5)
}

#points(df_train_pred$Date[df_train_pred$Model=="Linear"], 
#       df_train_pred$Value[df_train_pred$Model=="Linear"], col="steelblue", pch = 19, cex=0.5)
#points(df_test_pred$Date[df_test_pred$Model=="Linear"], 
#       df_test_pred$Value[df_test_pred$Model=="Linear"], col="brown", pch = 19, cex=0.5)

#points(df_train_pred$Date[df_train_pred$Model=="KNN"], 
#       df_train_pred$Value[df_train_pred$Model=="KNN"], col="steelblue", pch = 19, cex=0.5)
#points(df_test_pred$Date[df_test_pred$Model=="KNN"], 
#       df_test_pred$Value[df_test_pred$Model=="KNN"], col="brown", pch = 19, cex=0.5)

#points(df_train_pred$Date[df_train_pred$Model=="SVR"], 
#       df_train_pred$Value[df_train_pred$Model=="SVR"], col="steelblue", pch = 19, cex=0.5)
#points(df_test_pred$Date[df_test_pred$Model=="SVR"], 
#       df_test_pred$Value[df_test_pred$Model=="SVR"], col="brown", pch = 19, cex=0.5)

if(model_name_toplot=="Blended"){
  points(df_train_pred$Date[df_train_pred$Model=="Blended"], 
         df_train_pred$Value[df_train_pred$Model=="Blended"], col="steelblue", pch = 19, cex=0.5)
  points(df_test_pred$Date[df_test_pred$Model=="Blended"], 
         df_test_pred$Value[df_test_pred$Model=="Blended"], col="magenta4", pch = 19, cex=0.5)
}

#points(obs_df_plot$Date, obs_df_plot$Value)

text(x = min(df_train_pred$Date[df_train_pred$Model==model_name_toplot]), 
     y = max(df_train_pred$Value[df_train_pred$Model==model_name_toplot]-10),
     labels = metric_text_str, 
     pos = 4, col = "steelblue", cex= 0.8)

text(x = min(df_test_pred$Date[df_test_pred$Model==model_name_toplot]-100), 
     y = max(df_train_pred$Value[df_train_pred$Model==model_name_toplot]-10),
     labels = metric_text_str_test, 
     pos = 4, col = "magenta4", cex= 0.8)
dev.off()
