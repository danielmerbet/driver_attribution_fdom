library(randomForest)
library(Metrics)  # for rmse, if needed
library(lubridate)

set.seed(123)
case_study <- "feeagh" #sau or  feeagh
dir <- paste0("~/Documents/intoDBP/driver_attribution_fdom/",case_study, "/")
#load drivers (meteorology, soil,  streamflow and all possible variables)
data <- read.csv(paste0(dir, "data/data.csv"))
data$date <- as.Date(data$date)

#select only relevant drivers:
data <- data[c( "date", "sr","st28", "st100", "sm28", "sm100", "fdom", "doc_gwlf")]

#merge all and add julian day and random
data$cyday <- cos(yday(data$date)*pi/180)

# Sort data
#data <- data[order(data$date), ]

kge <- function(sim, obs) {
  r <- cor(sim, obs, use = "pairwise.complete.obs")
  alpha <- sd(sim, na.rm = TRUE) / sd(obs, na.rm = TRUE)
  beta <- mean(sim, na.rm = TRUE) / mean(obs, na.rm = TRUE)
  
  kge <- 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
  return(kge)
}

# Parameters
initial_train_size <- 500  # initial training size
test_window <- 100         # size of testing window
step_size <- 50            # step for moving window
tvar <- "fdom"

# Initialize results storage
results <- data.frame(
  Fold = integer(),
  TrainStart = as.Date(character()),
  TestStart = as.Date(character()),
  R2_train = double(), 
  NSE_train = double(), 
  KGE_train = double(),
  RMSE_train = double(),
  R2_test = double(), 
  NSE_test = double(),
  KGE_test = double(),
  RMSE_test = double()
)

fold <- 1
for (start in seq(initial_train_size, nrow(data) - test_window, by = step_size)) {
  
  train_idx <- 1:start
  test_idx <- (start + 1):(start + test_window)
  
  if (max(test_idx) > nrow(data)) break
  
  traindata <- data[train_idx, ]
  testdata <- data[test_idx, ]
  
  # Fit random forest
  formula <- as.formula(paste(tvar, "~ . - date"))
  RFfit <- randomForest(formula, data = traindata, ntree = 1000)
  
  # Predictions
  pred_train <- predict(RFfit, newdata = traindata)
  pred_test  <- predict(RFfit, newdata = testdata)
  
  obs_train <- traindata[[tvar]]
  obs_test  <- testdata[[tvar]]
  
  # Metrics
  r2_train  <- round(cor(pred_train, obs_train)^2, 2)
  nse_train <- round(1 - sum((obs_train - pred_train)^2) / sum((obs_train - mean(obs_train))^2), 2)
  kge_train <- round(kge(pred_train, obs_train), 2)
  rmse_train <- round(sqrt(mean((obs_train - pred_train)^2)), 2)
  
  r2_test  <- round(cor(pred_test, obs_test)^2, 2)
  nse_test <- round(1 - sum((obs_test - pred_test)^2) / sum((obs_test - mean(obs_test))^2), 2)
  kge_test <- round(kge(pred_test, obs_test), 2)
  rmse_test <- round(sqrt(mean((obs_test - pred_test)^2)), 2)
  
  # Store results
  results <- rbind(results, data.frame(
    Fold = fold,
    TrainStart = min(traindata$date),
    TestStart = min(testdata$date),
    R2_train = r2_train, 
    NSE_train = nse_train, 
    KGE_train = kge_train, 
    RMSE_train = rmse_train,
    R2_test = r2_test, 
    NSE_test = nse_test,
    KGE_test = kge_test,
    RMSE_test = rmse_test
  ))
  
  fold <- fold + 1
}

# Plot RMSE comparison
plot(results$Fold, results$RMSE_train, type='b', col='blue', ylim=range(c(results$RMSE_train, results$RMSE_test)),
     ylab="RMSE", xlab="Fold", main="Train vs. Test RMSE")
lines(results$Fold, results$RMSE_test, type='b', col='red')
legend("topright", legend=c("Train", "Test"), col=c("blue", "red"), lty=1)

# KGE comparison
plot(results$Fold, results$KGE_train, type='b', col='blue',ylim=c(0,1), # ylim=range(c(results$NSE_train, results$NSE_test)
     ylab="KGE", xlab="Fold", main="Train vs. Test KGE")
lines(results$Fold, results$KGE_test, type='b', col='red')
legend("bottomright", legend=c("Train", "Test"), col=c("blue", "red"), lty=1)
