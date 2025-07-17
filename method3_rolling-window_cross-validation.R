library(lubridate)
library(randomForest)

set.seed(123)
case_study <- "sau" #sau or  feeagh
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

# Define parameters
train_size <- 365      # 2 years
test_size <- 180       # 1 year
step_size <- 90        # shift window every 90 days

start_indices <- seq(1, nrow(data) - train_size - test_size, by = step_size)

results <- data.frame(Fold = integer(),
                      TrainStart = as.Date(character()),
                      TestStart = as.Date(character()),
                      R2 = numeric(),
                      NSE = numeric(),
                      RMSE = numeric(),
                      KGE = numeric(),
                      stringsAsFactors = FALSE)

for (k in seq_along(start_indices)) {
  start_train <- start_indices[k]
  end_train <- start_train + train_size - 1
  start_test <- end_train + 1
  end_test <- start_test + test_size - 1
  
  # Make sure we don't go beyond data
  if (end_test > nrow(data)) break
  
  train <- data[start_train:end_train, ]
  test <- data[start_test:end_test, ]
  
  RFfit <- randomForest(fdom ~ . - date, data = train, ntree = 500)
  pred <- predict(RFfit, test)
  
  obs <- test$fdom
  sim <- pred
  r2 <- round(cor(obs, sim)^2, 2)
  nse_val <- round(nse(sim, obs), 2)
  rmse_val <- round(sqrt(mean((sim - obs)^2)), 2)
  kge_val <- round(kge(sim, obs), 2)
  
  results[k, ] <- list(k, train$date[1], test$date[1], r2, nse_val, rmse_val, kge_val)
}

print(results)

library(ggplot2)

ggplot(results, aes(x = TestStart)) +
  geom_line(aes(y = KGE), color = "darkred") +
  geom_line(aes(y = R2), color = "steelblue") +
  ylab("Performance metric") + xlab("Test Start Date") +
  theme_minimal() +
  ggtitle("Rolling-window validation")
