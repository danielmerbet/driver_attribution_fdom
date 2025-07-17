library(lubridate)
library(randomForest)

set.seed(123)
case_study <- "sau" #sau or  feeagh
dir <- paste0("~/Documents/intoDBP/driver_attribution_fdom/",case_study, "/")
#load drivers (meteorology, soil,  streamflow and all possible variables)
data <- read.csv(paste0(dir, "data/data.csv"))
data$date <- as.Date(data$date)
best_predictor <- F
only_reanalysis <- F
plot_percent <- F

#select only relevant drivers:
if (best_predictor){
  if(case_study=="sau"){
    data <- data[c( "date", "v","st255", "sm100","sm255", "doc_gwlf", "fdom")]
  }else{
    data <- data[c( "date", "swt", "sr", "st100","st255", "sm100", "sm255", "doc_gwlf", "fdom")]
  }
}
if (only_reanalysis){
  if(case_study=="sau"){
    data <- data[c( "date", "st255", "sm100","sm255", "fdom")]
  }else{
    data <- data[c( "date", "sr", "st100","st255", "sm100", "sm255", "fdom")]
  }
}

#merge all and add julian day and random
data$cyday <- cos(yday(data$date)*pi/180)
#data$random <- runif(nrow(data))

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

set.seed(123)
tvar <- "fdom"
formula <- as.formula(paste(tvar, "~ . - date"))
RFfit <- randomForest(formula, data = traindata, ntree = 1000)

importance_random <- importance(RFfit); importance_random
importance_perc <- importance_random/sum(importance_random)*100
importance_perc
if(!(best_predictor & only_reanalysis)){
  write.csv(importance_perc, file = paste0(dir, "output/save_perc.csv"))
}

pred_oob <- predict(RFfit)
obs_train <- unlist(traindata[tvar])
rmse_oob <- sqrt(mean((obs_train - pred_oob)^2))
rsq_oob <- cor(obs_train, pred_oob)^2
nse_oob <- nse(pred_oob, obs_train)
kge_oob <- kge(pred_oob, obs_train)

pred_test <- predict(RFfit, newdata = testdata)
obs_test <- unlist(testdata[tvar])
rmse_test <- sqrt(mean(((obs_test) - pred_test)^2))
rsq_test <- cor(obs_test, pred_test)^2
nse_test <- nse(pred_test, obs_test)
kge_test <- kge(pred_test, obs_test)

cat("OOB R²:", round(rsq_oob, 2), "| OOB RMSE:", round(rmse_oob, 2), "\n")
cat("Test R²:", round(rsq_test, 2), "| Test RMSE:", round(rmse_test, 2), "\n")

add <- "all"
if (best_predictor){add <- "best"}
if (only_reanalysis){add <- "onlyreanalysis"}
pdf(paste0(dir, "output/method1_fixed_holdout_period_", add,".pdf"), width = 8, height = 6)
par(font.lab = 2) # Makes axis labels bold
plot(data$date, data[tvar][,1], ylab="fDOM (QSU)", 
     xlab = "", xaxt="n", bty="n", ylim=c(yi,ye))
axis.Date(1, at = seq(as.Date(paste0(year(min(data$date)),"-", "01-01")), 
                      as.Date(paste0(year(max(data$date)),"-", "12-31")),
                      by = "6 months"), format = "%m-%Y")
points(traindata$date, pred_oob, col="steelblue", pch = 19, cex=0.5)
points(testdata$date, pred_test, col="brown", pch = 19, cex=0.5)
#title(main="Optimal training period")
text(x = min(testdata$date)-40, y = max(data[tvar][,1]-5),
     labels = paste0("Testing metrics:",
                     "\nR² = ", round(rsq_test,2), 
                     "\nRMSE = ", round(rmse_test,2),
                     "\nKGE = ", round(kge_test,2)), 
     pos = 4, col = "brown")
text(x = min(traindata$date), y = max(data[tvar][,1]-5),
     labels = paste0("Out-of-bag metrics:",
                     "\nR² = ", round(rsq_oob,2), 
                     "\nRMSE = ", round(rmse_oob,2),
                     "\nKGE = ", round(kge_oob,2)), 
     pos = 4, col = "steelblue")
dev.off()

if(plot_percent){
  #Train on 30%, 50%, 70%, 85% and plot errors:
  fractions <- seq(0.1, 0.9, 0.1)
  train_errors <- c()
  test_errors <- c()
  
  for (f in fractions) {
    n_train <- round((n - n_holdout) * f)
    sub_train <- traindata[1:n_train, ]
    
    rf_tmp <- randomForest(formula, data = sub_train, ntree = 500)
    pred_tr <- predict(rf_tmp, newdata = sub_train)
    pred_te <- predict(rf_tmp, newdata = testdata)
    
    train_errors <- c(train_errors, sqrt(mean((sub_train$fdom - pred_tr)^2)))
    test_errors <- c(test_errors, sqrt(mean((testdata$fdom - pred_te)^2)))
  }
  
  add <- "trainpercentage"
  pdf(paste0(dir, "output/method1_fixed_holdout_period_", add,".pdf"), width = 8, height = 6)
  plot(fractions, train_errors, type = "o", col = "blue", ylim = range(c(train_errors, test_errors)),
       ylab = "RMSE", xlab = "Training Fraction", main = "Learning Curve")
  lines(fractions, test_errors, type = "o", col = "red")
  legend("topright", legend = c("Training", "Testing"), col = c("blue", "red"), lty = 1)
  dev.off()
}