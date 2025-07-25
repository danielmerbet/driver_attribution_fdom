
#LOAD DATA
library(lubridate)
library(dplyr)

case_study <- "sau" #feeagh or sau
dir <- paste0("~/Documents/intoDBP/driver_attribution_fdom/", case_study, "/")

# Load data
data <- read.csv(paste0(dir, "data/data.csv"))
data$date <- as.Date(data$date)

# Add cos of julian day
data$cyday <- cos(yday(data$date) * pi / 180)

# Select columns
#if (case_study == "sau") {
#  data <- data[, c("v", "st255", "sm100", "sm255", "doc_gwlf", "cyday", "fdom", "date")]
#}
#if (case_study == "feeagh") {
#  data <- data[, c("swt", "sr", "st100", "st255", "sm100", "sm255", "doc_gwlf", "cyday", "fdom", "date")]
#}

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

#MACHINE LEARNING METHODS
library(randomForest)
set.seed(123)
rf <- randomForest(fdom ~ . -date, data = train_data, ntree = 1000)
pred_rf <- predict(rf, newdata = test_data)
write.csv(importance(rf)/sum(importance(rf)), 
          file=paste0(dir,"output/importance/imp_rf.csv"),
          quote = F)

library(xgboost)
set.seed(123)
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test))

params <- list(objective = "reg:squarederror", eta = 0.1, max_depth = 5)
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 500)
pred_xgb <- predict(xgb_model, newdata = dtest)
xgb.importance(model = xgb_model)
write.csv(xgb.importance(model = xgb_model), 
          file=paste0(dir,"output/importance/imp_xgb.csv"),
          quote = F)

library(lightgbm)
set.seed(123)
lgb_train <- lgb.Dataset(data = as.matrix(X_train), label = y_train)
lgb_model <- lgb.train(params = list(objective = "regression", learning_rate = 0.1, metric = "rmse"), 
                       data = lgb_train, nrounds = 500)
pred_lgb <- predict(lgb_model, newdata = as.matrix(X_test))
lgb.importance(lgb_model)
write.csv(lgb.importance(lgb_model), 
          file=paste0(dir,"output/importance/imp_lgb.csv"),
          quote = F)

library(catboost)
set.seed(123)
train_pool <- catboost.load_pool(data = as.matrix(X_train), label = y_train)
test_pool <- catboost.load_pool(data = as.matrix(X_test))

cat_model <- catboost.train(learn_pool = train_pool, 
                            params = list(iterations = 500, learning_rate = 0.1, depth = 6))
pred_cat <- catboost.predict(cat_model, pool = test_pool)

data.frame(var=colnames(X_train),
           importance=catboost.get_feature_importance(cat_model)/sum(catboost.get_feature_importance(cat_model)))

write.csv(data.frame(var=colnames(X_train),
                     importance=catboost.get_feature_importance(cat_model)/sum(catboost.get_feature_importance(cat_model))), 
          file=paste0(dir,"output/importance/imp_cbt.csv"),
          quote = F)


#liner regresion
set.seed(123)
lm_model <- lm(fdom ~ ., data = train_data)
pred_lm <- predict(lm_model, newdata = test_data)
#doesn't directly provide feature importance scores

library(kknn)
set.seed(123)
knn_model <- kknn::train.kknn(fdom ~ ., data = train_data, kmax = 5)
pred_knn <- predict(knn_model, newdata = test_data)
#doesn't directly provide feature importance scores

library(kernlab)
set.seed(123)
svr_model <- ksvm(fdom ~ ., data = train_data, kernel = "rbfdot", C = 1)
pred_svr <- predict(svr_model, newdata = test_data)
#doesn't directly provide feature importance scores


#stack models:
library(caret)
library(Metrics)

# Train models separately, then combine
preds <- cbind(pred_rf, pred_xgb, pred_lgb, pred_cat, pred_lm)
meta_model <- lm(y_test ~ preds)
pred_stack <- predict(meta_model)

#print all results form each model
results <- list()
set.seed(123)
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
evaluate("CatBoost", pred_cat)
evaluate("Linear", pred_lm)
evaluate("KNN", pred_knn)
evaluate("SVR", pred_svr)
evaluate("Stacked", pred_stack)

results_df <- do.call(rbind.data.frame, results)
print(results_df)

#PLOT ALL MODELS
library(ggplot2)
library(grid)

# Store training and test predictions
preds_training <- cbind(
  predict(rf, newdata = train_data),
  catboost.predict(cat_model, pool = train_pool),
  predict(lm_model, newdata = train_data)
)
meta_model_training <- lm(y_train ~ preds_training)
train_predictions_list <- list(
  RF = predict(rf, newdata = train_data),
  XGB = predict(xgb_model, newdata = xgb.DMatrix(data = as.matrix(X_train))),
  LGB = predict(lgb_model, newdata = as.matrix(X_train)),
  CatBoost = catboost.predict(cat_model, pool = train_pool),
  Linear = predict(lm_model, newdata = train_data),
  KNN = predict(knn_model, newdata = train_data),
  SVR = predict(svr_model, newdata = train_data),
  Stacked = predict(meta_model_training)
)

test_predictions_list <- list(
  RF = pred_rf,
  XGB = pred_xgb,
  LGB = pred_lgb,
  CatBoost = pred_cat,
  Linear = pred_lm,
  KNN = pred_knn,
  SVR = pred_svr,
  Stacked = pred_stack
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

plot_comparison_all_models_with_train_test <- function(
    dates_train, y_train_actual, train_predictions_list,
    dates_test, y_test_actual, test_predictions_list,
    train_metrics_list, test_metrics_list,
    model_colors = NULL,
    title = "Model Comparison (Train + Test)",
    filename = "model_comparison_train_test.pdf"
) {
  
  # Combine predictions into a data frame
  all_train <- lapply(names(train_predictions_list), function(model_name) {
    data.frame(
      Date = dates_train,
      Value = train_predictions_list[[model_name]],
      Model = model_name,
      Period = "Train"
    )
  })
  df_train_pred <- do.call(rbind, all_train)
  df_train_actual <- data.frame(Date = dates_train, Value = y_train_actual, Model = "Actual", Period = "Train")
  
  all_test <- lapply(names(test_predictions_list), function(model_name) {
    data.frame(
      Date = dates_test,
      Value = test_predictions_list[[model_name]],
      Model = model_name,
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
  if (is.null(model_colors)) {
    model_colors <- setNames(
      c("black", rainbow(length(train_predictions_list))),
      c("Actual", names(train_predictions_list))
    )
  }
  
  # Plot
  p <- ggplot(df_plot, aes(x = Date, y = Value, color = Model, linetype = Period)) +
    geom_line(size = 1) +
    scale_color_manual(values = model_colors) +
    labs(title = title, y = "fdom", color = "Model", linetype = "Period") +
    theme_minimal() +
    theme(legend.position = "right")
  
  # Format metrics for display
  metric_text <- unlist(lapply(names(test_metrics_list), function(model_name) {
    train_metrics <- train_metrics_list[[model_name]]
    test_metrics <- test_metrics_list[[model_name]]
    paste0(model_name, ": | Train: R2=", sprintf("%.2f", train_metrics$R2),
           " KGE=", sprintf("%.2f", train_metrics$KGE),
           " | Test: R2=", sprintf("%.2f", test_metrics$R2),
           " KGE=", sprintf("%.2f", test_metrics$KGE))
  }))
  
  metric_text_str <- paste(metric_text, collapse = "\n")
  metric_grob <- grid::textGrob(metric_text_str, x = 0.1, y = 0.9, just = c("left", "top"), hjust = 0, vjust = 1, gp = grid::gpar(fontsize = 9))
  
  # Save to PDF
  pdf(filename, width = 16, height = 6)
  print(p)
  grid.draw(metric_grob)
  dev.off()
}

# Call the plotting function
plot_comparison_all_models_with_train_test(
  dates_train = train_data$date,
  y_train_actual = y_train,
  train_predictions_list = train_predictions_list,
  
  dates_test = test_data$date,
  y_test_actual = y_test,
  test_predictions_list = test_predictions_list,
  
  train_metrics_list = train_metrics_list,
  test_metrics_list = test_metrics_list,
  
  #title = "Supervised Machine Learning Approaches (Train + Test)",
  title = "",
  filename = paste0(dir, "output/model_comparison_train_test_all-feaures.pdf")
)






#BASURA
#plot all results
library(ggplot2)

plot_comparison <- function(dates, actual, pred, title, filename) {
  df <- data.frame(Date = dates, Actual = actual, Predicted = pred)
  
  p <- ggplot(df, aes(x = Date)) +
    geom_line(aes(y = Actual, color = "Actual")) +
    geom_line(aes(y = Predicted, color = "Predicted")) +
    labs(title = title, y = "fdom", color = "Legend") +
    theme_minimal() +
    theme(legend.position = "top")
  
  ggsave(filename, plot = p, width = 10, height = 5, dpi = 300)
}

plot_comparison(test_data$date, y_test, pred_rf, "Random Forest Prediction", "rf_plot.pdf")
plot_comparison(test_data$date, y_test, pred_xgb, "XGBoost Prediction", "xgb_plot.pdf")
plot_comparison(test_data$date, y_test, pred_rf, "Random Forest Prediction", "rf_plot.pdf")
plot_comparison(test_data$date, y_test, pred_stack, "Stacked", "stacked_plot.pdf")


#FIRST TRY FOR ALL MODELS

library(ggplot2)
library(grid)

plot_comparison_all_models <- function(
    dates_test, y_test_actual, predictions_list,
    metrics_list, model_colors = NULL,
    title = "Model Comparison (Test Period)", 
    filename = "model_comparison_all.pdf"
) {
  # Create a data frame with all predictions
  all_preds <- lapply(names(predictions_list), function(model_name) {
    data.frame(
      Date = dates_test,
      Value = predictions_list[[model_name]],
      Model = model_name
    )
  })
  
  df_pred <- do.call(rbind, all_preds)
  df_actual <- data.frame(Date = dates_test, Value = y_test_actual, Model = "Actual")
  
  # Combine actual and predictions
  df_plot <- rbind(df_pred, df_actual)
  
  # Default colors if not provided
  if (is.null(model_colors)) {
    model_colors <- setNames(
      c("black", rainbow(length(predictions_list))),
      c("Actual", names(predictions_list))
    )
  }
  
  # Plot
  p <- ggplot(df_plot, aes(x = Date, y = Value, color = Model, linetype = Model)) +
    geom_line(size = 1) +
    scale_color_manual(values = model_colors) +
    scale_linetype_manual(values = rep("solid", length(model_colors))) +
    labs(title = title, y = "fdom", color = "Model", linetype = "Model") +
    theme_minimal() +
    theme(legend.position = "right")
  
  # Format metrics for display
  metric_text <- unlist(lapply(names(metrics_list), function(model_name) {
    metrics <- metrics_list[[model_name]]
    paste0(model_name, ": R2 = ", sprintf("%.2f", metrics$R2), 
           ", RMSE = ", sprintf("%.2f", metrics$RMSE),
           ", NSE = ", sprintf("%.2f", metrics$NSE),
           ", KGE = ", sprintf("%.2f", metrics$KGE))
  }))
  
  # Add metrics as text (wrap if needed)
  metric_text_str <- paste(metric_text, collapse = "\n")
  metric_grob <- grid::textGrob(metric_text_str, x = 0, y = 1, just = c("left", "top"), hjust = 0, vjust = 1)
  
  # Save plot with metrics text
  pdf(filename, width = 14, height = 6)
  print(p)
  grid.draw(metric_grob)
  dev.off()
}

# Store predictions in a list
predictions_list <- list(
  RF = pred_rf,
  XGB = pred_xgb,
  LGB = pred_lgb,
  CatBoost = pred_cat,
  Linear = pred_lm,
  KNN = pred_knn,
  SVR = pred_svr,
  Stacked = pred_stack
)

# Store metrics in a list
metrics_list <- list()
for (model_name in names(predictions_list)) {
  metrics_list[[model_name]] <- list(
    R2 = r2(y_test, predictions_list[[model_name]]),
    RMSE = rmse(y_test, predictions_list[[model_name]]),
    NSE = nse(y_test, predictions_list[[model_name]]),
    KGE = kge(y_test, predictions_list[[model_name]])
  )
}

# Call the plotting function
plot_comparison_all_models(
  dates_test = test_data$date,
  y_test_actual = y_test,
  predictions_list = predictions_list,
  metrics_list = metrics_list,
  title = "Model Comparison on Test Period",
  filename = "model_comparison_all.pdf"
)
