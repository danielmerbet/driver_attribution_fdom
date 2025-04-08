library(lubridate)

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

#ML Analysis
###############################################################
#Random forest
library(randomForest)
if (case_study=="sau"){
  i <- 433 #best fitting 
  yi<-0;ye<-60
}
if (case_study=="feeagh"){
  i <- 101 #best fitting 
  yi<-40;ye<-90
}
train_perc <- 0.85 #percentage for training 
training_number <- round(dim(data)[1]*train_perc)
total_front <- dim(data)[1]-training_number
number_test <- dim(data)[1]-total_front
m <- (1+i):(1+i+total_front)
traindata <- data[-m,]
testdata <- data[m,]

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

pdf(paste0(dir, "output/simulation1.pdf"), width = 8, height = 6)
par(font.lab = 2) # Makes axis labels bold
plot(data$date, data[tvar][,1], ylab="fDOM (QSU)", 
     xlab = "", xaxt="n", bty="n", ylim=c(yi,ye))
axis.Date(1, at = seq(as.Date(paste0(year(min(data$date)),"-", "01-01")), 
                      as.Date(paste0(year(max(data$date)),"-", "12-31")),
                      by = "6 months"), format = "%m-%Y")
points(traindata$date, predRF_OOB, col="steelblue", pch = 19, cex=0.5)
points(testdata$date, predRF, col="brown", pch = 19, cex=0.5)
#title(main="Optimal training period")
text(x = min(traindata$date), y = max(data[tvar][,1]-10),
     labels = paste0("R² = ", rsq_test, "\nNSE = ", nse_test, "\nRMSE = ", rmse_test), 
     pos = 4, col = "black")
dev.off()

plot(testdata[tvar][,1],predRF, xlab="Obs", ylab="Sim", ylim=c(5,52),xlim=c(5,52))
abline(0,1, col="red")

####Plotting nude purity in bars
name_variable = names(data)[!names(data) %in% c("fdom", "date")]

#png(paste0(dir, "output/NudePurityBars.png"), width = 800, height = 600, bg = NA)
# Adjust plot margins to make space for longer variable names
#par(mar = c(5, 15, 4, 2))  # Increase the left margin (second value)

# Create the bar plot with better spacing for variable names
barplot(
  height = rev(importance_perc),
  names.arg = rev(name_variable),
  horiz = TRUE,
  col = "steelblue",
  las = 1,  # Rotate labels for better readability
  cex.names = 1,  # Reduce font size of variable names
  main = "Sau Reservoir",
  xlab = "Nude purity (%)",
  ylab = "Predictor"
)
#dev.off()

barplot_table <- data.frame(name_variable=name_variable, 
                            importance_perc=importance_perc)

write.csv(barplot_table, paste0(dir, "output/barplot_table.csv"), quote = F, row.names = F)
