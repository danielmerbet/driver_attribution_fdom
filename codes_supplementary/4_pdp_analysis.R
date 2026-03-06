library(lubridate); library(pdp)
library(gridExtra) # for arranging multiple plots
library(ggplot2)
library(here)

setwd(paste0(dirname(rstudioapi::getSourceEditorContext()$path), "/../"))

set.seed(123)
case_study <- "feeagh" #feeaghor sau
dir <- paste0(getwd(),"/",case_study, "/")
#load drivers (meteorology, soil,  streamflow and all possible variables)
data <- read.csv(paste0(dir, "data/data.csv"))
data$date <- as.Date(data$date)

#merge all and add julian day and random
data$cyday <- cos(yday(data$date)*pi/180)
#data$random <- runif(nrow(data))

#select best parameters
#data <- data[,c("v", "st255","sm100", "sm255","doc_gwlf", "cyday", "fdom", "date")]

r2 <- function(obs, pred) {
  cor(obs, pred)^2
}

# RMSE
rmse <- function(obs, pred) {
  sqrt(mean((obs - pred)^2))
}

# NSE
nse <- function(obs, pred) {
  numerator <- sum((obs - pred)^2)
  denominator <- sum((obs - mean(obs))^2)
  1 - (numerator / denominator)
}

# KGE
kge <- function(obs, pred) {
  r <- cor(pred, obs, use = "pairwise.complete.obs")
  alpha <- sd(pred, na.rm = TRUE) / sd(obs, na.rm = TRUE)
  beta <- mean(pred, na.rm = TRUE) / mean(obs, na.rm = TRUE)
  1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
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
set.seed(123)
tvar <- "fdom"
formula <- as.formula(paste(tvar, "~ . - date"))
if (case_study=="sau"){
  RFfit <- randomForest(formula, data = traindata, ntree = 300, mtry=2)
}
if (case_study=="feeagh"){
  RFfit <- randomForest(formula, data = traindata, ntree = 300, mtry=6)
}

#plot(RFfit)

#Partial dependence plots

#ggplot nice
#Partial dependence plots
# Define the variables
if (case_study=="sau"){
  vars <- c("v", "st255", "sm100", "sm255", "cyday") #"doc_gwlf"
  yi<-10; ye<-18
}

if (case_study=="feeagh"){
  vars <- c("swt", "sr", "st100","st255", "sm100", "sm255", "doc_gwlf", "cyday")
  yi<-57; ye<-63
}

# Generate the plots
plots <- lapply(vars, function(var) {
  ggplot(partial(RFfit, pred.var = var), aes_string(x = var, y = "yhat")) +
    geom_point() +
    geom_smooth(span = 0.2) +
    theme_bw() +
    #theme_minimal() +  # Minimal theme (removes grey background)
    theme(
      panel.border = element_blank(),  # Remove the black box around the plot
      panel.grid.major = element_blank(),  # Remove major gridlines
      panel.grid.minor = element_blank()   # Remove minor gridlines
    ) +
    theme(axis.line = element_line(colour = "black"))+
    ylim(yi,ye)+
    labs(x = var, y = "Avg. fDOM predicted")
})

# Save to PDF
pdf(paste0(dir, "output/partial_dependence_plots.pdf"), width = 7, height = 7)
if (case_study=="sau"){
  grid.arrange(grobs = plots, ncol = 2, nrow = 3)
}
if (case_study=="feeagh"){
  grid.arrange(grobs = plots, ncol = 2, nrow = 4)
}
dev.off()
