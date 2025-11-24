
#LOAD DATA
library(lubridate)
library(dplyr)
library(DEoptim)

#RUN supervised_machine-learning_selected-features.R
#ltes optimised belnding for the three best models
# Objective function: minimize RMSE
blend_rmse <- function(weights) {
  w1 <- weights[1]
  w2 <- weights[2]
  w3 <- 1 - w1 - w2
  
  # Penalize if weights are outside 0–1 range
  if (w1 < 0 || w2 < 0 || w3 < 0 || w1 > 1 || w2 > 1 || w3 > 1) return(1e6)
  
  if (case_study=="sau"){
    blended <- w1 * pred_lm + w2 * pred_rf + w3 * pred_ctb
  }else{
    blended <- w1 * pred_lm + w2 * pred_rf + w3 * pred_ctb
  }
  
  
  # Penalize if predictions are not finite
  if (any(!is.finite(blended))) return(1e6)
  
  return(rmse(y_test, blended))
}

# Initial weights: equal parts
initial_weights <- c(1/3, 1/3)

# Optimize
opt <- optim(
  par = initial_weights,
  fn = blend_rmse,
  method = "L-BFGS-B",
  lower = c(0, 0),
  upper = c(1, 1)
)

# Extract optimal weights
w1 <- opt$par[1]
w2 <- opt$par[2]
w3 <- 1 - w1 - w2
cat("Optimal weights:\n")

if (case_study=="sau"){
  cat("Linear:   ", round(w1, 3), "\n")
  cat("RF:       ", round(w2, 3), "\n")
  cat("CatBoost: ", round(w3, 3), "\n")
}else{
  cat("Linear:       ", round(w1, 3), "\n")
  cat("RF:       ", round(w2, 3), "\n")
  cat("CatBoost: ", round(w3, 3), "\n")
}


# Final blended prediction
if (case_study=="sau"){
  blended_preds <- w1 * pred_lm + w2 * pred_rf + w3 * pred_ctb
}else{
  blended_preds <- w1 * pred_lm + w2 * pred_rf + w3 * pred_ctb
}

evaluate("Blended1", blended_preds)

# Evaluate the final blended model
#results <- data.frame(
#  R2 = cor(y_test, blended_preds)^2,
#  RMSE = rmse(y_test, blended_preds),
#  NSE = NSE(blended_preds, y_test),
#  KGE = KGE(blended_preds, y_test)
#)
#print(results)

#Let's use all of the models:
# Your 7-model prediction matrix
pred_matrix <- cbind(pred_rf, pred_xgb, pred_lgb, pred_ctb) #, pred_lm, pred_knn, pred_svr

# Objective function: minimize RMSE
blend_rmse_de <- function(weights) {
  weights <- weights / sum(weights)  # Normalize to sum to 1
  blended <- pred_matrix %*% weights
  if (any(is.na(blended))) return(1e6)
  return(rmse(y_test, blended))
}

n_models <- ncol(pred_matrix)

set.seed(123)
opt_result <- DEoptim(
  fn = blend_rmse_de,
  lower = rep(0, n_models),
  upper = rep(1, n_models),
  DEoptim.control(
    NP = 50,
    itermax = 200,
    trace = TRUE
  )
)

opt_weights <- opt_result$optim$bestmem
opt_weights <- opt_weights / sum(opt_weights)
names(opt_weights) <- c("RF", "XGB", "LGB", "CatBoost") #, "Linear""KNN", "SVR"
print(round(opt_weights, 3))

# Final blended prediction
blended_pred2 <- pred_matrix %*% opt_weights

evaluate("Blended2", blended_pred2)

results_df <- do.call(rbind.data.frame, results)
print(results_df)
