#This is an exploration of time series analysis that includes moving average, holt-winters smoothing, and ARIMA models.
---
title: "Explore Time Series Analysis"
output: html_notebook
---
#loading libraries
library(data.table)
library(dplyr)
library(TTR)
library(forecast)

#Read file
smc_test <- fread('smc_test.csv', stringsAsFactors = FALSE)


smc_test <- arrange(smc_test, usage_date,cust_loc_id)



smc <- smc_test %>%  group_by(usage_date) %>% summarise(usage_ccf = sum(usage_ccf)/1000)
smc_ts_matrix <- as.matrix(smc[2])

smc_timeseries <- ts(smc_ts_matrix, frequency=12, start=c(2014,1))
#plotting the timeseries
plot.ts(smc_timeseries)

#moving average of second order
smoothedsmc <- SMA(smc_timeseries,n=2)
#we can observe that theres a severe downfall at mid 2015
plot.ts(smoothedsmc)


#here we can observe trend and seasonal components separately
smccomponents <- decompose(smc_timeseries)
plot(smccomponents)

#to detach the seasonal component
smcseasonallyadjusted <- smc_timeseries - smccomponents$seasonal
plot(smcseasonallyadjusted)

#simple exponential smoothing for default ts assuming no trend
smc_notrend <- HoltWinters(smc_timeseries, beta=FALSE, gamma=FALSE)
smc_notrend
smc_notrend$SSE
smc_notrend$fitted
#In plot, Red line is the forecast, Black line is the original data
plot(smc_notrend)



#simple exponential smoothing for seasonally adjusted ts assuming no trend
smc_adjusted <- HoltWinters(smcseasonallyadjusted, beta=FALSE, gamma=FALSE)
smc_adjusted

smc_adjusted$fitted
#plotting seasonally adjusted data forecast
plot(smc_adjusted)

#gives us slightly better sum of squared errors
smc_adjusted$SSE

#predicting for next 5 months
smc_adjusted_forecasts <- forecast.HoltWinters(smc_adjusted, h=5)
smc_adjusted_forecasts

plot.forecast(smc_adjusted_forecasts)

plot(smc_adjusted_forecasts$residuals)

#No non-zero auto correlations - p value is high
acf(smc_adjusted_forecasts$residuals, lag.max = 20,na.action = na.pass)
Box.test(smc_adjusted_forecasts$residuals, lag=20, type="Ljung-Box")

#Function to check whether forecast errors are normally distributed with mean zero
plotForecastErrors <- function(forecasterrors)
{
  # make a histogram of the forecast errors:
  mybinsize <- IQR(forecasterrors,na.rm=TRUE)/4
  mysd   <- sd(forecasterrors,na.rm=TRUE)
  mymin  <- min(forecasterrors,na.rm=TRUE) - mysd*5
  mymax  <- max(forecasterrors,na.rm=TRUE) + mysd*3
  # generate normally distributed data with mean 0 and standard deviation mysd
  mynorm <- rnorm(10000, mean=0, sd=mysd)
  mymin2 <- min(mynorm)
  mymax2 <- max(mynorm)
  if (mymin2 < mymin) { mymin <- mymin2 }
  if (mymax2 > mymax) { mymax <- mymax2 }
  # make a red histogram of the forecast errors, with the normally distributed data overlaid:
  mybins <- seq(mymin, mymax, mybinsize)
  hist(forecasterrors, col="red", freq=FALSE, breaks=mybins)
  # freq=FALSE ensures the area under the histogram = 1
  # generate normally distributed data with mean 0 and standard deviation mysd
  myhist <- hist(mynorm, plot=FALSE, breaks=mybins)
  # plot the normal curve as a blue line on top of the histogram of forecast errors:
  points(myhist$mids, myhist$density, type="l", col="blue", lwd=2)
}

# The plot shows that the distribution of forecast errors is roughly centred on zero, and is more or less normally
# distributed.
# The Ljung-Box test showed that there is little evidence of non-zero autocorrelations in the in-sample forecast errors,
# and the distribution of forecast errors seems to be normally distributed with mean zero. This suggests that the
# simple exponential smoothing method provides an adequate predictive model for water usage. Furthermore, the assumptions that the 80% and 95% predictions intervals were based upon
# (that there are no autocorrelations in the forecast errors, and the forecast errors are normally distributed with
# mean zero and constant variance) are probably valid.
plotForecastErrors(smc_adjusted_forecasts$residuals)


#Holts exponential model(assuming there is a trend and no seasonality)
smc_trend <- HoltWinters(smc_timeseries, gamma=FALSE)
smc_trend

smc_trend$SSE

plot(smc_trend)

#lets give starting values for level and slope b of trend component (diff of first and second values)
smc_trend <- HoltWinters(smc_timeseries, gamma=FALSE, l.start=0.536, b.start=0.145)
smc_trend

smc_trend$SSE

plot(smc_trend)

#lets predict for the next 5 months
smc_trend <- forecast.HoltWinters(smc_trend, h=5)
plot.forecast(smc_trend)

#p-value is very low - indicates evidence of non-zero auto correlations. Model could be improved.
acf(smc_trend$residuals, lag.max = 20,na.action = na.pass)
Box.test(smc_trend$residuals, lag=20, type="Ljung-Box")

#lets check for constant variance and normal distribution with mean zero in residuals
plot.ts(smc_trend$residuals)            # make a time plot
plotForecastErrors(smc_trend$residuals) # make a histogram
#If we observe carefully, mean is to the right of zero. This model shoud not be considered.
mean(smc_trend$residuals,na.rm=TRUE)

#Holt-Winter's Exponential smoothing (assuming there is trend and seasonality)
smc_trend_seasonality <- HoltWinters(smc_timeseries)

smc_trend_seasonality$SSE

plot(smc_trend_seasonality)

#predicting for the next 5 months
smc_trend_seasonality <- forecast.HoltWinters(smc_trend_seasonality, h=5)
smc_trend_seasonality
plot(smc_trend_seasonality)

#p-value is high , residuals are independent
acf(smc_trend_seasonality$residuals, lag.max=20,na.action = na.pass)
Box.test(smc_trend_seasonality$residuals, lag=20, type="Ljung-Box")

plot.ts(smc_trend_seasonality$residuals)            # make a time plot
plotForecastErrors(smc_trend_seasonality$residuals) # make a histogram


#ARIMA Model
#allows for non-zero auto correlations to exist

acf(smc_timeseries, lag.max=20)             # plot a correlogram
acf(smc_timeseries, lag.max=20, plot=FALSE) # get the autocorrelation values

pacf(smc_timeseries, lag.max=20)
pacf(smc_timeseries, lag.max=20, plot=FALSE)

#we can select a model(p,d,q) based on the above information or go with a nice function auto.arima() to select optimal
#model

#passing original timeseries without any adjustments or differences. Auto Arima takes care of differences and selects
#optimal model
smctrain <- auto.arima(smc_ts_matrix, trace = TRUE)
smctrain

smcprediction <- forecast.Arima(smctrain, h=5)
smcprediction
plot.forecast(smcprediction)

acf(smcprediction$residuals, lag.max=20)
Box.test(smcprediction$residuals, lag=20, type="Ljung-Box")

plot.ts(smcprediction$residuals)            # make time plot of forecast errors
plotForecastErrors(smcprediction$residuals) # make a histogram
mean(smcprediction$residuals)

# The successive forecast errors do not seem to be correlated, but the forecast errors do not seem to be normally 
# distributed with mean zero and constant variance, the ARIMA(1,1,0) does not seem to be an adequate predictive 
# model for the water usage and could be improved. 

