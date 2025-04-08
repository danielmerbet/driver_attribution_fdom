library(lubridate)

set.seed(123)
case_study <- "feeagh"
dir <- paste0("~/Documents/intoDBP/driver_attribution_fdom/",case_study, "/")
#load drivers (meteorology, soil,  streamflow and all possible variables)
data <- read.csv(paste0(dir, "data/data_old.csv"))
data$date <- as.Date(data$date, format = "%d/%m/%Y")

data <- na.omit(data)

write.csv(data, file=paste0(dir, "data/data.csv"), 
          quote=F, row.names = F)
