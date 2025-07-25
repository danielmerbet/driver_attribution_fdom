library(lubridate)

set.seed(123)
case_study <- "sau" #sau or  feeagh
dir <- paste0("~/Documents/intoDBP/driver_attribution_fdom/",case_study, "/")
#load drivers (meteorology, soil,  streamflow and all possible variables)
data <- read.csv(paste0(dir, "data/data.csv"))
data$date <- as.Date(data$date)

#merge all and add julian day and random
data$cyday <- cos(yday(data$date)*pi/180)
data$random <- runif(nrow(data))

#selected parameters
if (case_study=="sau"){
  data <- data[,c("v", "st255","sm100", "sm255","doc_gwlf", "cyday", "fdom", "date")]
} 
if (case_study=="feeagh"){
  data <- data[,c("swt", "sr","st100","st255","sm100", "sm255","doc_gwlf", "cyday", "fdom", "date")]
} 

nse <- function(sim, obs) {
  numerator <- sum((obs - sim)^2)
  denominator <- sum((obs - mean(obs))^2)
  nse <- 1 - (numerator / denominator)
  return(nse)
}

kge <- function(sim, obs) {
  r <- cor(sim, obs, use = "pairwise.complete.obs")
  alpha <- sd(sim, na.rm = TRUE) / sd(obs, na.rm = TRUE)
  beta <- mean(sim, na.rm = TRUE) / mean(obs, na.rm = TRUE)
  
  kge <- 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
  return(kge)
}

#ML Analysis
###############################################################
#Random forest
library(randomForest)
if (case_study=="sau"){
  yi<-0;ye<-60
}
if (case_study=="feeagh"){
  yi<-40;ye<-90
}
# Define fixed holdout (e.g., last 15%)
n <- nrow(data)
holdout_ratio <- 0.15
n_holdout <- round(n * holdout_ratio)

# Training: first 85%, Testing: last 15%
traindata <- data[1:(n - n_holdout), ]
testdata <- data[(n - n_holdout + 1):n, ]

#start training 
tvar <- "fdom"
set.seed(123)
formula <- as.formula(paste(tvar, "~ . - date"))
RFfit <- randomForest(formula, data = traindata, ntree = 1000)
#plot(RFfit)

#check resulting stats with OOB data
set.seed(123)
predRF_OOB<- predict(RFfit) # without data, give the prediction with OOB samples
rsqOOB <- round((cor(predRF_OOB, traindata[tvar]))^2,2) ; rsqOOB
rmseOOB <- round(sqrt(mean((traindata[tvar][,1] - predRF_OOB)^2)), 2); rmseOOB
maeOOB <- mean(abs(traindata[tvar][,1] - predRF_OOB));maeOOB
importance_random <- importance(RFfit); importance_random
importance_perc <- importance_random/sum(importance_random)*100
importance_perc

#testing: check with data not used in training 
set.seed(123)
predRF<- predict(RFfit, testdata) # without data, give the prediction with OOB samples
rsq_test <- round((cor(predRF, testdata[tvar]))^2,2) ; rsq_test
rmse_test <- round(sqrt(mean((testdata[tvar][,1] - predRF)^2)), 2); rmse_test
nse_test <- round(nse(testdata[tvar][,1], predRF),2); nse_test
importance_random <- importance(RFfit); importance_random
importance_perc <- importance_random/sum(importance_random)*100
importance_perc

##############33
#LSTM
# Get RF predictions on both training and testing data
predRF_train <- predict(RFfit, traindata)
predRF_test <- predict(RFfit, testdata) # You already have this as 'predRF'

# Calculate the residuals (the part the RF model couldn't explain)
residuals_train <- traindata$fdom - predRF_train
residuals_test <- testdata$fdom - predRF_test

##arrange data for LSTM
library(keras)
# Hyperparameter: Define the look-back window for the LSTM
look_back <- 10 # Use the last 10 residuals to predict the next one

# --- Data Normalization ---
# Create a simple min-max scaler function
scaler <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

unscaler <- function(x, min_val, max_val) {
  return (x * (max_val - min_val) + min_val)
}

# IMPORTANT: Fit the scaler ONLY on the training data's residuals
min_resid_train <- min(residuals_train)
max_resid_train <- max(residuals_train)

scaled_residuals_train <- scaler(residuals_train)
# Apply the SAME scaling to the test data
scaled_residuals_test <- (residuals_test - min_resid_train) / (max_resid_train - min_resid_train)


# --- Create sequences for the LSTM ---
# Function to restructure data into [samples, timesteps, features]
create_sequences <- function(data, look_back) {
  x <- list()
  y <- list()
  for (i in 1:(length(data) - look_back)) {
    x[[i]] <- data[i:(i + look_back - 1)]
    y[[i]] <- data[i + look_back]
  }
  return(list(x = array(unlist(x), dim = c(length(x), look_back, 1)), y = unlist(y)))
}

# Create training and testing sequences
train_sequences <- create_sequences(scaled_residuals_train, look_back)
test_sequences <- create_sequences(scaled_residuals_test, look_back)

X_train_lstm <- train_sequences$x
y_train_lstm <- train_sequences$y

X_test_lstm <- test_sequences$x
y_test_lstm <- test_sequences$y

y_train_lstm <- as.matrix(y_train_lstm)

#Implement LSTM
# Clear any previous Keras session
# Clear any previous Keras session
k_clear_session()
set.seed(123) # For reproducibility of LSTM training

# Define the LSTM model architecture using a list (more robust syntax)
lstm_model <- keras_model_sequential(list(
  layer_lstm(units = 50, input_shape = c(look_back, 1), return_sequences = TRUE),
  layer_dropout(rate = 0.2),
  layer_lstm(units = 50),
  layer_dropout(rate = 0.2),
  layer_dense(units = 1)
))

# Compile the model
lstm_model$compile(
  loss = 'mean_squared_error',
  optimizer = 'adam'
)

summary(lstm_model)

# Train the model
history <- lstm_model$fit(
  X_train_lstm, y_train_lstm,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.1,
  verbose = 2
)

plot(history)

pdf(paste0(dir, "output/simulation2.pdf"), width = 8, height = 6)
par(font.lab = 2) # Makes axis labels bold
plot(data$date, data[tvar][,1], ylab="fDOM (QSU)", 
     xlab = "", xaxt="n", bty="n", ylim=c(yi,ye))
axis.Date(1, at = seq(as.Date(paste0(year(min(data$date)),"-", "01-01")), 
                      as.Date(paste0(year(max(data$date)),"-", "12-31")),
                      by = "6 months"), format = "%m-%Y")
points(traindata$date, predRF_OOB, col="steelblue", pch = 19, cex=0.5)
points(testdata$date, predRF, col="brown", pch = 19, cex=0.5)
#title(main="Optimal training period with best predictors")
text(x = min(traindata$date), y = max(data[tvar][,1]-10),
     labels = paste0("R² = ", rsq_test, "\nNSE = ", nse_test, "\nRMSE = ", rmse_test), pos = 4, col = "black")
dev.off()

plot(testdata[tvar][,1],predRF, xlab="Obs", ylab="Sim", ylim=c(5,52),xlim=c(5,52))
abline(0,1, col="red")
