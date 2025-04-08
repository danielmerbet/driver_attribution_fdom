library(lubridate); library(pdp)
library(gridExtra) # for arranging multiple plots
library(ggplot2)

set.seed(123)
case_study <- "feeagh" #or sau
dir <- paste0("~/Documents/intoDBP/driver_attribution_fdom/",case_study, "/")
#load drivers (meteorology, soil,  streamflow and all possible variables)
data <- read.csv(paste0(dir, "data/data.csv"))
data$date <- as.Date(data$date)

#merge all and add julian day and random
data$cyday <- cos(yday(data$date)*pi/180)
data$random <- runif(nrow(data))

#select best parameters
#data <- data[,c("v", "st255","sm100", "sm255","doc_gwlf", "cyday", "fdom", "date")]

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
}
if (case_study=="feeagh"){
  i <- 101 #best fitting 
}
train_perc <- 0.85 #percentage for training 
training_number <- round(dim(data)[1]*train_perc)
total_front <- dim(data)[1]-training_number
number_test <- dim(data)[1]-total_front
m <- (1+i):(1+i+total_front)
traindata <- data[-m,]
testdata <- data[m,]

#start training 
set.seed(123)
tvar <- "fdom"
formula <- as.formula(paste(tvar, "~ . - date"))
RFfit <- randomForest(formula, data = traindata, ntree = 1000)
#plot(RFfit)

#Partial dependence plots
#partial(RFfit, pred.var = var, plot = T, plot.engine = "ggplot2", rug=T)
var <- "st255"
partial(RFfit, pred.var = var)
var <- "sm100"
partial(RFfit, pred.var = var)
var <- "sm255"
partial(RFfit, pred.var = var)
var <- "doc_gwlf"
partial(RFfit, pred.var = var)
var <- "cyday"
partial(RFfit, pred.var = var)

#ggplot nice
#Partial dependence plots
# Define the variables
if (case_study=="sau"){
  vars <- c("v", "st255", "sm100", "sm255", "doc_gwlf", "cyday")
  yi<-12; ye<-16
}

if (case_study=="feeagh"){
  vars <- c("swt", "sr", "st100","st255", "sm100", "sm255", "doc_gwlf", "cyday")
  yi<-58; ye<-64
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


