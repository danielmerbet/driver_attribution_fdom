library(keras)
library(tensorflow)
library(dplyr)
library(caret)

# Set case study and load data
case_study <- "feeagh" #"sau"
dir <- paste0("~/Documents/intoDBP/driver_attribution_fdom/", case_study, "/")

X_train <- read.csv(paste0(dir, "output/shap/X_train.csv"))
X_test <- read.csv(paste0(dir, "output/shap/X_test.csv"))
y_train <- read.csv(paste0(dir, "output/shap/y_train.csv")) %>% unlist() %>% as.numeric()
y_test  <- read.csv(paste0(dir, "output/shap/y_test.csv")) %>% unlist() %>% as.numeric()

# Scale features
X_train_scaled <- scale(X_train)
X_test_scaled <- scale(X_test,
  center = attr(X_train_scaled, "scaled:center"),
  scale = attr(X_train_scaled, "scaled:scale")
)

# Scale target
y_mean <- mean(y_train)
y_sd <- sd(y_train)
y_train_scaled <- (y_train - y_mean) / y_sd
y_test_scaled  <- (y_test  - y_mean) / y_sd

# Create supervised dataset
create_lagged_dataset <- function(X, y, lag = 30) {
  X_array <- array(NA, dim = c(nrow(X) - lag, lag, ncol(X)))
  y_array <- y[(lag + 1):length(y)]
  for (i in 1:(nrow(X) - lag)) {
    X_array[i,,] <- X[i:(i + lag - 1), ]
  }
  list(X = X_array, y = y_array)
}

lag <- 7
train_supervised <- create_lagged_dataset(X_train_scaled, y_train_scaled, lag = lag)
test_supervised  <- create_lagged_dataset(X_test_scaled, y_test_scaled, lag = lag)

# Define simpler LSTM model builder
build_model <- function(units = 64, dropout = 0.1, lr = 0.001) {
  model <- keras_model_sequential() %>%
    layer_lstm(units = units, input_shape = c(lag, ncol(X_train_scaled)), return_sequences = FALSE) %>%
    layer_dropout(rate = dropout) %>%
    layer_dense(units = 1)

  model %>% compile(
    loss = "mean_squared_error",
    optimizer = optimizer_adam(learning_rate = lr),
    metrics = list("mean_squared_error")
  )
  return(model)
}

# Hyperparameter tuning loop
units_list <- c(32, 64)
dropout_list <- c(0, 0.1)
lr_list <- c(0.0005, 0.001)
epochs <- 100
batch_size <- 16

results <- data.frame()
best_mse <- Inf

for (units in units_list) {
  for (dropout in dropout_list) {
    for (lr in lr_list) {
      cat("Training model: units =", units, "dropout =", dropout, "lr =", lr, "\n")
      model <- build_model(units, dropout, lr)

      history <- model %>% fit(
        x = train_supervised$X, y = train_supervised$y,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = 0.2,
        callbacks = list(callback_early_stopping(patience = 10, restore_best_weights = TRUE)),
        verbose = 0
      )

      plot(history, main = paste("Training History (units=", units, ", dropout=", dropout, ", lr=", lr, ")"))

      val_mse <- min(history$metrics$val_mean_squared_error)
      results <- rbind(results, data.frame(units, dropout, lr, val_mse))

      if (val_mse < best_mse) {
        best_model <- model
        best_mse <- val_mse
      }
    }
  }
}

# Predict and inverse-scale target
pred_lstm_scaled <- predict(best_model, test_supervised$X)
obs_lstm_scaled  <- test_supervised$y

# Rescale to original units
pred_lstm <- pred_lstm_scaled * y_sd + y_mean
obs_lstm  <- obs_lstm_scaled  * y_sd + y_mean

# Metrics
r2 <- function(obs, pred) cor(obs, pred)^2
rmse <- function(obs, pred) sqrt(mean((obs - pred)^2))
nse <- function(obs, pred) 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)

cat("Improved LSTM Results:\n")
cat("R2:", round(r2(obs_lstm, pred_lstm), 2), "\n")
cat("RMSE:", round(rmse(obs_lstm, pred_lstm), 2), "\n")
cat("NSE:", round(nse(obs_lstm, pred_lstm), 2), "\n")

# Save the best model
save_model(best_model, filepath = paste0(dir, "output/models/lstm_best_model.keras"))

# Optional: plot observed vs predicted
plot(obs_lstm, type = "l", col = "black", lwd = 2,
     ylab = "fDOM", xlab = "Time Index", main = "LSTM: Observed vs Predicted")
lines(pred_lstm, col = "red", lwd = 2)
legend("topright", legend = c("Observed", "Predicted"),
       col = c("black", "red"), lwd = 2)

