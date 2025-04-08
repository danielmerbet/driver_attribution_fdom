# driver_attribution_fdom

STEP 1: Create your onw data, see example in feaagh/data/ and sau/data

STEP 2: Use best_training-testing.R to shift the testing period among the time series, to assess the fitting for all possible combinations of training and testing periods. 

STEP 3: Use purity_predictors.R and to get the selected predictors, and pdp_analysis.R to obtain the partial dependant plot

STEP 4: use simulation1.R, simulation2.R, and simulation3.R, to run with all predictors, with the predictors with high (>5%) increase in node purity, and only using predictors extracted from reanalysis data
 
