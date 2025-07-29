library(keras)
library(tensorflow)
library(dplyr)
library(caret)

#it only worked running in terminal, in my case

case_study <- "sau" #feeaghor sau
dir <- paste0("~/Documents/intoDBP/driver_attribution_fdom/",case_study, "/")

#load data
X_train <- read.csv(paste0(dir,"output/shap/X_train.csv"))
X_test <- read.csv(paste0(dir,"output/shap/X_test.csv"))
y_train <- read.csv(paste0(dir,"output/shap/y_train.csv"))
y_train <- as.numeric(unlist(y_train))
y_test <- read.csv(paste0(dir,"output/shap/y_test.csv"))
y_test <- as.numeric(unlist(y_test))

#Prepara time series
# Scale features
X_train_scaled <- scale(X_train)
X_test_scaled <- scale(X_test, center = attr(X_train_scaled, "scaled:center"),
                       scale = attr(X_train_scaled, "scaled:scale"))

# Function to convert to supervised learning format (lagged features)
create_lagged_dataset <- function(X, y, lag = 5) {
  X_array <- array(NA, dim = c(nrow(X) - lag, lag, ncol(X)))
  y_array <- y[(lag + 1):length(y)]
  for (i in 1:(nrow(X) - lag)) {
    X_array[i,,] <- X[i:(i + lag - 1), ]
  }
  list(X = X_array, y = y_array)
}

lag <- 10  # can be tuned later
train_supervised <- create_lagged_dataset(X_train_scaled, y_train, lag = lag)
test_supervised <- create_lagged_dataset(X_test_scaled, y_test, lag = lag)

#LSTM model fucntion
build_model <- function(units = 64, dropout = 0.2, lr = 0.001) {
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

#hyperparameter tuning
# Define grid
units_list <- c(32, 64)
dropout_list <- c(0.2, 0.4)
lr_list <- c(0.001, 0.01)
epochs <- 50
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
        epochs = epochs, batch_size = batch_size,
        validation_split = 0.15, verbose = 0
      )
      
      val_mse <- tail(history$metrics$val_mean_squared_error, 1)
      
      results <- rbind(results, data.frame(units, dropout, lr, val_mse))
      
      if (val_mse < best_mse) {
        best_model <- model
        best_mse <- val_mse
      }
    }
  }
}

#make predictions:
pred_lstm <- predict(best_model, test_supervised$X)
obs_lstm <- test_supervised$y

# Metrics
r2 <- function(obs, pred) cor(obs, pred)^2
rmse <- function(obs, pred) sqrt(mean((obs - pred)^2))
nse <- function(obs, pred) 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)

cat("LSTM Results:\n")
cat("R2:", round(r2(obs_lstm, pred_lstm), 2), "\n")
cat("RMSE:", round(rmse(obs_lstm, pred_lstm), 2), "\n")
cat("NSE:", round(nse(obs_lstm, pred_lstm), 2), "\n")

save_model_hdf5(best_model, filepath = paste0(dir, "output/models/lstm_best_model.h5"))