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
create_lagged_dataset <- function(X, y, lag = 5) {
  X_array <- array(NA, dim = c(nrow(X) - lag, lag, ncol(X)))
  y_array <- y[(lag + 1):length(y)]
  for (i in 1:(nrow(X) - lag)) {
    X_array[i,,] <- X[i:(i + lag - 1), ]
  }
  list(X = X_array, y = y_array)
}
lag <- 10  # can be tuned later
test_supervised <- create_lagged_dataset(X_test_scaled, y_test, lag = lag)

#load previously ynied model
lstm_model <- load_model_hdf5(filepath = paste0(dir, "output/models/lstm_best_model.h5"))

#make prediction
pred_lstm <- predict(lstm_model, test_supervised$X)
obs_lstm <- test_supervised$y  # true target values aligned with prediction


pdf(paste0(dir,"output/lstm.pdf"))
plot(obs_lstm, type = "l", col = "black", lwd = 2,
     ylab = "fDOM", xlab = "Time Index", main = "LSTM: Observed vs Predicted")
lines(pred_lstm, col = "red", lwd = 2)
legend("topright", legend = c("Observed", "Predicted"),
       col = c("black", "red"), lwd = 2)
dev.off()
