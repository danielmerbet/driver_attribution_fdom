library(lubridate)

set.seed(123)
case_study <- "feeagh" #or sau
dir <- paste0("~/Documents/intoDBP/driver_attribution_fdom/",case_study, "/")
#load drivers (meteorology, soil,  streamflow and all possible variables)
data <- read.csv(paste0(dir, "data/data.csv"))
data$date <- as.Date(data$date)

#merge all and add julian day and random
data$cyday <- cos(yday(data$date)*pi/180)
data$random <- runif(nrow(data))

#correlation plot
data <- data[,2:ncol(data)]
numeric_data <- data[, sapply(data, is.numeric)]
#compute the correlation matrix
cor_matrix <- cor(numeric_data, use = "complete.obs")
#plot the correlation matrix
library(corrplot)
pdf(paste0(dir, "output/correlation.pdf"), width = 7, height = 7)
corrplot(cor_matrix, method = "circle", type="upper")
dev.off()
