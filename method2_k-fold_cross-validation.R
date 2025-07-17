library(lubridate)
library(randomForest)

set.seed(123)
case_study <- "feeagh" #sau or  feeagh
dir <- paste0("~/Documents/intoDBP/driver_attribution_fdom/",case_study, "/")
#load drivers (meteorology, soil,  streamflow and all possible variables)
data <- read.csv(paste0(dir, "data/data.csv"))
data$date <- as.Date(data$date)

#merge all and add julian day and random
data$cyday <- cos(yday(data$date)*pi/180)
data$random <- runif(nrow(data))

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

# Number of folds
k <- 5 #it correspond to the 15% data in testing, about 1 month
fold_size <- floor(nrow(data) / (k + 1))  # +1 to leave space for final test

# Initialize results
results <- data.frame(
  Fold = integer(),
  R2 = numeric(),
  NSE = numeric(),
  RMSE = numeric()
)

#start plot:
pdf(paste0(dir, "output/method2_fold_cross_validation.pdf"), width = 10, height = 6)
plot(data$date, data$fdom, ylab="fDOM (QSU)", 
     xlab = "", xaxt="n", bty="n", ylim=c(yi,ye))
axis.Date(1, at = seq(as.Date(paste0(year(min(data$date)),"-", "01-01")), 
                      as.Date(paste0(year(max(data$date)),"-", "12-31")),
                      by = "6 months"), format = "%m-%Y")

# Loop through time-based folds
for (fold in 1:k) {
  # Define training and testing indices
  train_end <- fold * fold_size
  test_start <- train_end + 1
  test_end <- test_start + fold_size - 1
  
  if (test_end > nrow(data)) break  # avoid overflow
  
  # Create training and testing sets
  traindata <- data[1:train_end, ]
  testdata <- data[test_start:test_end, ]
  
  # Fit random forest
  formula <- as.formula("fdom ~ . - date")
  RFfit <- randomForest(formula, data = traindata, ntree = 1000)
  
  #predict training data
  pred_oob <- predict(RFfit)
  obs_train <- unlist(traindata$fdom)
  rmse_oob <- sqrt(mean((obs_train - pred_oob)^2))
  rsq_oob <- cor(obs_train, pred_oob)^2
  nse_oob <- nse(pred_oob, obs_train)
  kge_oob <- kge(pred_oob, obs_train)
  
  #predict on test data
  pred <- predict(RFfit, testdata)
  r2 <- round((cor(pred, testdata$fdom))^2, 2)
  rmse <- round(sqrt(mean((testdata$fdom - pred)^2)), 2)
  nse_val <- round(nse(testdata$fdom, pred), 2)
  kge_val <- round(kge(testdata$fdom, pred), 2)
  
  # Save results
  results <- rbind(results, data.frame(Fold = fold, R2 = r2, 
                                       NSE = nse_val, RMSE = rmse, KGE=kge_val))
  
  abline(v=testdata$date[1]-1,lwd=2)
  points(testdata$date, pred, col="brown", pch = 19, cex=0.5)
  #title(main="Optimal training period")
  text(x = min(testdata$date), y = max(data$fdom-5),
       labels = paste0("K fold: ", fold), 
       pos = 4, col = "black")
  text(x = min(testdata$date), y = max(data$fdom-20),
       labels = paste0("Out-of-bag:",
                       "\nR² = ", round(rsq_oob,2), 
                       "\nRMSE = ", round(rmse_oob,2),
                       "\nKGE = ", round(kge_oob,2)), 
       pos = 4, col = "steelblue")
  text(x = min(testdata$date), y = max(data$fdom-35),
       labels = paste0("Testing:",
                       "\nR² = ", round(r2,2), 
                       "\nRMSE = ", round(rmse,2),
                       "\nKGE = ", round(kge_val,2)), 
       pos = 4, col = "brown")
}
dev.off()
print(results)
