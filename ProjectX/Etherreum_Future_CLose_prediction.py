import numpy as np
import pandas as pd
import os
from subprocess import check_output
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
import statsmodels.tsa.holtwinters as ets
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from pandas.plotting import lag_plot
import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from helper import  ADF_Cal, kpss_test, Cal_rolling_mean_var, calculate_MA, plot_detrended, LSE,estimated_variance,calculate_gpac,inverse_diff,plot_gpac
import os
warnings.filterwarnings('ignore')

#loading data
df = pd.read_csv('ETH_1H.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
#data pre-processing
print(df.index)
ID=pd.Series(range(0,len(df.Close)))
print(df.columns)
df.set_index(ID,inplace=True)
print(df.index)

print(df.isnull().sum().sort_values(ascending=True))
# no null value found

print(df.nunique(dropna=False))
#converting the Date to day, month, year.
df['Date'] = pd.to_datetime(df['Date'])
df['Month']=df['Date'].dt.month
df['Day']=df['Date'].dt.day
df['Year']=df['Date'].dt.year
df['Hour']=df['Date'].dt.hour
df['Minute']=df['Date'].dt.minute
df['Second']=df['Date'].dt.second
# Data Analysis and Visualization
# Market Capitalization:
# The total dollar market value of a company's outstanding shares of stock is
# referred to as market capitalization. It is computed by multiplying the entire
# number of a company's outstanding shares by the current market price of one share,
# which is commonly referred to as "market cap."
df['Market_cap']= df['Open']*df['Volume']

print(df.iloc[df['Market_cap'].argmax()])

# Volatility : In order to KNOW the volatility of the stock,
# we find daily percentage change in the closing price of the stock

df['volatility'] = (df['Close']/df['Close'].shift(1)) - 1

# ploting Month vs High, low, Open, close, volume

def plot_Date_Vs_features(dataset):
        for i in dataset:
            if i == 'Open' or i == 'High' or i == 'Low' or i == 'Close' or i == 'Market_cap':
                plt.figure(figsize=(20,8))
                plt.plot(dataset['Date'],dataset[i], label=i, color= 'orange')
                plt.title(f'Date vs {i}',fontsize='20')
                plt.xlabel('Date',fontsize='20')
                plt.ylabel(f'price in {i} USD',fontsize='20')
                plt.grid()
                plt.legend()
                plt.show()
#Plot volume
fig, ax = plt.subplots(figsize=(20,8))
ax.plot(df['Date'], df['Volume'], color='blue')
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
ax.set_xlabel('Date', fontsize='20')
ax.set_ylabel('Volume', fontsize='20')
plt.title('Volume Sold per Hour',fontsize='20')
plt.grid()
plt.show()
#plot volatility
#In order to know the volatility of the stock, we find the daily percentage change
# in the closing price of the stock.
fig, ax = plt.subplots(figsize=(20,8))
ax.plot(df['Date'], df['volatility'], color='blue')
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
ax.set_xlabel('Date', fontsize='20')
ax.set_ylabel('volatility', fontsize='20')
plt.title('Change in Closing price',fontsize='20')
plt.grid()
plt.show()
# histogram of Volatility
df['volatility'].hist(bins=100, color='blue');

#Cumulative return
# A cumulative return on an investment is the total
# amount gained or lost by the investment throughout time,
# regardless of the length of time involved.

df['Cumulative Return'] = (1 + df['volatility']).cumprod()
fig, ax = plt.subplots(figsize=(20,8))
ax.plot(df['Date'], df['Cumulative Return'], color='red')
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
ax.set_xlabel('Date', fontsize='20')
ax.set_ylabel('Cumulative Return', fontsize='20')
plt.title('Cumulative Return', fontsize='20')
plt.grid()
plt.show()


x=plot_Date_Vs_features(df)
print(x)

# OPEN, CLOSE, HIGH, LOW OF ETH
fig, ax = plt.subplots(figsize=(20,8))
ax.plot(df['Date'], df['Open'], color='Red', label='Open')
ax.plot(df['Date'], df['Close'], color='Green',label='Close')
ax.plot(df['Date'], df['High'], color='Blue', label='High')
ax.plot(df['Date'], df['Low'], color='Yellow', label='Low')
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
ax.set_xlabel('Date', fontsize='20')
ax.set_ylabel('Price in USD', fontsize='20')
plt.title('ETH Prices', fontsize='20')
plt.grid()
plt.legend()
plt.show()

#Rolling mean and rolling variance
Cal_rolling_mean_var(df.Date,df.Open,'Open')
Cal_rolling_mean_var(df.Date,df.Open,'High')
Cal_rolling_mean_var(df.Date,df.Open,'Low')
Cal_rolling_mean_var(df.Date,df.Open,'Close')
Cal_rolling_mean_var(df.Date,df.Open,'Volume')


# check Dependent variable is stationary or not  Via ADF and Kpss test

print('\n')
print('ADF test of Close Price')
ADF_Cal(df['Close'])
print('\n')
print('KPSS test of Close Price')
kpss_test(df['Close'])
print('\n')

# As p value is greater than 0.05 and ADF statistics are greater than critical values( 1%, 5% and 10%) we can say that the Close price is non-stationary
# As p value is smaller than 0.05 and Test statistics are greater than critical values( 1%, 2.5%, 5% and 10%) we can say that the Close price is non-stationary

# Log Transformation and 1st order differencing to make the Close price stationary

df["Close_LogT"]= np.log(df['Close'])
def differencing(dataset):
    diff_close=[]
    for i in range(0,len(dataset)):
        if i==0:
            diff_close.append(df.Close_LogT[0])
        else:
            value= dataset[i]-dataset[i-1]
            diff_close.append(value)
    return diff_close

df["Differential_close"]=differencing(df.Close_LogT)
print('\n')
print('ADF test of Differential_Close Price')
ADF_Cal(df["Differential_close"])
print('\n')
print('KPSS test of Differential_Close Price')
kpss_test(df["Differential_close"])
print('\n')

#differencing made the dependent variable Stationary as per ADF and Kpss test

#ACF and PACF plots before and After differencing
# Before
plot_acf(df['Close'])
plt.xlabel('Lags')
plt.ylabel('ACF of Close')
plt.legend()
plt.show()
plot_pacf(df['Close'], lags=50)
plt.xlabel('Lags')
plt.ylabel('PACF of CLose')
plt.legend()
plt.show()

#After

plot_acf(df["Differential_close"],label ='After Differencing')
plt.xlabel('Lags')
plt.ylabel('ACF of CLose')
plt.legend()
plt.show()
plot_pacf(df["Differential_close"], lags=50,label ='After Differencing')
plt.xlabel('Lags')
plt.ylabel('PACF of CLose')
plt.legend()
plt.show()

#rolling mean and Rolling variance
#before
Cal_rolling_mean_var(df['Date'],df['Close'],'Close')
#After
Cal_rolling_mean_var(df.Date,df["Differential_close"],'Differential_close')

# #Moving average
MA3_df = calculate_MA(df['Close'], 3, 1, False)
detrended = df['Close'].to_numpy() / MA3_df

plot_detrended(df.Close[0:50], MA3_df[0:50], detrended[0:50], 'Date', 'Close',
               'Eth 3 Months Moving Average', '3-MA')

MA5_df = calculate_MA(df['Close'], 5, 1, False)
detrended = df['Close'].to_numpy() / MA5_df
plot_detrended(df.Close[0:50], MA5_df[0:50], detrended[0:50], 'Time', 'AirPassenger',
               'ETH 5 Months Moving Average', '5-MA')

MA7_df = calculate_MA(df['Close'], 7, 1, False)
detrended = df['Close'].to_numpy() / MA7_df
plot_detrended(df.Close[0:50], MA7_df[0:50], detrended[0:50], 'Time', 'AirPassenger',
               'ETH 7 Months Moving Average', '7-MA')

MA9_df = calculate_MA(df['Close'], 9, 1, False)
detrended = df['Close'].to_numpy() / MA9_df
plot_detrended(df.Close[0:50], MA9_df[0:50], detrended[0:50], 'Time', 'AirPassenger',
               'ETH 9 Months Moving Average', '9-MA')
# Using the function developed in the step 1 plot the estimated cycle-trend versus the original dataset
# (plot only the first 50 samples) for 2x4-MA, 2x6-MA, 2x8-MA, and 2x10-MA. Plot the detrended data on
# the same graph. Add an appropriate title, x-label, y-label, and legend to the graph.

MA4_df = calculate_MA(df['Close'], 4, 2, False)
detrended = df['Close'].to_numpy() / MA4_df

plot_detrended(df.Close[0:50], MA4_df[0:50], detrended[0:50], 'Time', 'AirPassenger',
               'ETH 2 x 4 Months Moving Average', '2 x 4-MA')

MA6_df = calculate_MA(df['Close'], 6, 2, False)
detrended = df['Close'].to_numpy() / MA6_df
plot_detrended(df.Close[0:50], MA6_df[0:50], detrended[0:50], 'Time', 'AirPassenger',
               'ETH 2 x 6 Months Moving Average', '2 x 6-MA')

MA8_df = calculate_MA(df['Close'], 8, 2, False)
detrended = df['Close'].to_numpy() / MA8_df
plot_detrended(df.Close[0:50], MA8_df[0:50], detrended[0:50], 'Time', 'AirPassenger',
               'ETH 2 x 8 Months Moving Average', '2 x 8-MA')

MA10_df = calculate_MA(df['Close'], 10, 2, False)
detrended = df['Close'].to_numpy() / MA10_df
plot_detrended(df.Close[0:50], MA10_df[0:50], detrended[0:50], 'Time', 'AirPassenger',
               'ETH 2 x 10 Months Moving Average', '2 x 10-MA')

# Compare the ADF-test of the original dataset versus the detrended dataset using the 3-MA
text = (df['Close'].values)
print(text)

print("----- MA3------------")
detrended = df['Close'].to_numpy() / MA3_df
text = ADF_Cal(detrended[1:-1])
print(text)

#  Apply the STL decomposition method to the dataset. Plot the trend, seasonality, and reminder in one
# graph. Add an appropriate title, x-label, y-label, and legend to the graph.

plt.figure(figsize=[10, 10])
# STL = STL(data, seasonal=13)
res = STL(df['Close'], period=24)

res = res.fit()
fig = res.plot()

T = res.trend
S = res.seasonal
R = res.resid

plt.figure(figsize=[8, 5])
plt.plot(T[:50], label='trend')
plt.plot(S[:50], label='Seasonal')
plt.plot(R[:50], label='residuals')
plt.title('Time series decomposition')
plt.xlabel('Time')
plt.ylabel('Close')
plt.legend()
plt.show()

# Calculate the seasonally adjusted data and plot it versus the original data.
# Add an appropriate title, xlabel, y-label, and legend to the graph

seasonal_adjust = df['Close'] - res.seasonal

plt.figure(figsize=[8, 5])
plt.plot_date(df.Date, df['Close'], ls='solid', c='green', label='original', marker='')
plt.plot_date(df.Date, seasonal_adjust, ls='solid', c='red', label='Seasonal Adjusted', marker='')
plt.xlabel('Time')
plt.ylabel('Close')
plt.title('Seasonal adjusted data for Close price')
plt.legend()
plt.show()

# Calculate the strength of trend using the following equation and display
# the following message on the console
F_t = max(0, (1 - (np.var(res.resid) / (np.var(np.array(res.trend + res.resid))))))

print("The strength of trend for this data set is {}".format(F_t))

#  Calculate the strength of seasonality using the following equation and display
# the following message on the console:

F_t = max(0, (1 - (np.var(res.resid) / (np.var(res.seasonal + res.resid)))))

print("The strength of seasonality for this data set is {}".format(F_t))

#Using the Holt-Winters method try to find the best fit using the train dataset and make a prediction using the test set.

#spliting the data.
y=df['Close']
yt,yf=train_test_split(y,shuffle=False,test_size=0.3)
lags=2000

def MSE(orignal,predicted):
    error=orignal-predicted
    error2=error**2
    return np.mean(error2)

def var_error(orignal,predicted):
    error=orignal-predicted
    return np.var(error)

def rolling_cal1(series):
    dummy_mean=[]
    for i in range(len(series)):
        dummy_mean.append(np.mean( series.head(i) ))
    return dummy_mean

def naive_method(series):
    naive=[]
    for i in range(len(series)):
        if i==0:
            naive.append(np.nan)
        else:
            naive.append(series[i-1])
    return naive

def drift_onestep(series):
    drift=[]
    h=1
    for i in range(len(series)):
        if i<=1:
            drift.append(np.nan)
        else:
            drift.append(series[i-1]+h*((series[i-1]-series[0])/(i-1)))
    return drift


def auto_correlation_calculator(series, lags):
    y = np.array(series).copy()
    y_mean = np.mean(series)
    cor = []
    for lag in np.arange(0, lags + 1):
        if lag==0:
            cor.append(1)
        else:
            num1 = y[lag:] - y_mean
            num2 = y[:-lag] - y_mean
            num = sum(num1 * num2)
            den = sum((y - y_mean) ** 2)
            cor.append(num / den)
    return pd.Series(cor)

def Qvalue(series,lags):
    r=auto_correlation_calculator(series,lags)
    rk=r**2
    return len(series)*(np.sum(rk))

def correlation_coefficent_cal(x,y):
    x=np.array(x)
    y=np.array(y)
    numerator=(np.sum((x-np.mean(x))*(y-np.mean(y))))
    denominator= np.sqrt(np.sum((x-np.mean(x))**2)) * np.sqrt(np.sum((y-np.mean(y))**2))
    r=numerator/denominator
    return r


averagef=[]
for i in range(len(yf)):
    averagef.append(np.mean(yt))


averaget=rolling_cal1(yt)


fig,ax=plt.subplots()
ax.plot(yt,label='Train Data')
ax.plot(yf,label='Test Data')
ax.plot(np.arange(len(yt),len(yt)+len(yf)),averagef,label='Average Forecast')
ax.set_title('Average method forecast')
ax.set_xlabel('time')
ax.set_ylabel('data')
ax.legend()
plt.show()

average_forecast_mse=MSE(yf,averagef)
print(f'MSE of average method :{average_forecast_mse}')
average_var_pred_error=var_error(yt,averaget)
print(f'variance of prediction error of average method:{average_var_pred_error}')
average_var_forecast_error=var_error(yf,averagef)
print(f'variance of forecast error of average method:{average_var_forecast_error}')


error_average=yf-averagef
acf_average = auto_correlation_calculator(error_average, lags)

plt.stem(np.arange(-lags, lags + 1), np.hstack(((acf_average[::-1])[:-1],acf_average)), linefmt='grey', markerfmt='o')
# plt.stem(np.arange(1, lags + 1), acf_average, linefmt='grey', markerfmt='o')
plt.title("ACF Plot of Average Forecast Errors")
plt.xlabel("Lags")
plt.ylabel("ACF")
plt.grid()
plt.legend(["ACF"], loc='lower right')
plt.show()

average_qvalue=Qvalue(averagef,lags)
print(f'qvalue of forecast of average method:{average_qvalue}')

average_corr=correlation_coefficent_cal(error_average,yf)
print(f'correlation coefficient of forecast error and test set of average method:{average_corr}')

# naive method
print("---------------Naive method----------------")
naivet=naive_method(yt)

naivef=[]
for i in range(len(yf)):
    naivef.append(yt[len(yt)-1])

fig,ax=plt.subplots()
ax.plot(yt,label='Train Data')
ax.plot(yf,label='Test Data')
ax.plot(np.arange(len(yt),len(yt)+len(yf)),naivef,label='Naive Forecast')
ax.set_title('Naive method forecast')
ax.set_xlabel('time')
ax.set_ylabel('data')
ax.legend()
plt.show()

naive_forecast_mse=MSE(yf,naivef)
print(f'MSE of naive method :{naive_forecast_mse}')
naive_var_pred_error=var_error(yt,naivet)
print(f'variance of prediction error of naive method:{naive_var_pred_error}')
naive_var_forecast_error=var_error(yf,naivef)
print(f'variance of forecast error of naive method:{naive_var_forecast_error}')

error_naive=yf-naivef
acf_naive = auto_correlation_calculator(error_naive, lags)

plt.stem(np.arange(-lags, lags + 1), np.hstack(((acf_naive[::-1])[:-1],acf_naive)), linefmt='grey', markerfmt='o')
# plt.stem(np.arange(1, lags + 1), acf_naive, linefmt='grey', markerfmt='o')
plt.title("ACF Plot of Naive Forecast Errors")
plt.xlabel("Lags")
plt.ylabel("ACF")
plt.grid()
plt.legend(["ACF"], loc='lower right')
plt.show()

naive_qvalue=Qvalue(naivef,lags)
print(f'qvalue of forecast of naive method:{naive_qvalue}')

naive_corr=correlation_coefficent_cal(error_naive,yf)
print(f'correlation coefficient of forecast error and test set of naive method:{naive_corr}')


# drift method

print("---------------Drift method----------------")
driftt=drift_onestep(yt)


driftf=[]
h=len(yf)
for i in range(1,h+1):
    driftf.append(yt[len(yt)-1]+i*((yt[len(yt)-1]-yt[0])/(len(yt)-1)))

fig,ax=plt.subplots()
ax.plot(yt,label='Train Data')
ax.plot(yf,label='Test Data')
ax.plot(np.arange(len(yt),len(yt)+len(yf)),driftf,label='Drift Forecast')
ax.set_title('Drift method forecast')
ax.set_xlabel('time')
ax.set_ylabel('data')
ax.legend()
plt.show()

drift_forecast_mse=MSE(yf,driftf)
print(f'MSE of drift method :{drift_forecast_mse}')
drift_var_pred_error=var_error(yt[2:],driftt[2:])
print(f'variance of prediction error of drift method:{drift_var_pred_error}')
drift_var_forecast_error=var_error(yf,driftf)
print(f'variance of forecast error of drift method:{drift_var_forecast_error}')

error_drift=yf-driftf
acf_drift = auto_correlation_calculator(error_drift, lags)

plt.stem(np.arange(-lags, lags + 1), np.hstack(((acf_drift[::-1])[:-1],acf_drift)), linefmt='grey', markerfmt='o')
# plt.stem(np.arange(1, lags + 1), acf_drift, linefmt='grey', markerfmt='o')
plt.title("ACF Plot of Drift Forecast Errors")
plt.xlabel("Lags")
plt.ylabel("ACF")
plt.grid()
plt.legend(["ACF"], loc='lower right')
plt.show()


drift_qvalue=Qvalue(driftf,lags)
print(f'qvalue of forecast of drift method:{drift_qvalue}')

drift_corr=correlation_coefficent_cal(error_drift,yf)
print(f'correlation coefficient of forecast error and test set of drift method:{drift_corr}')


# SES
print("---------------SES method----------------")
ses_holtt=ets.ExponentialSmoothing(yt,trend=None,damped_trend=False,seasonal=None).fit()
ses_holtf=ses_holtt.forecast(steps=len(yf))
ses_holtf=pd.DataFrame(ses_holtf).set_index(yf.index)[0]

ses_prediction = ets.ExponentialSmoothing(yt, trend=None, damped_trend=False, seasonal=None).fit(smoothing_level=0.5)
ses_prediction_model = ses_prediction.forecast(steps=len(yt))
ses_predictions = pd.DataFrame(ses_prediction_model).set_index(yt.index)[0]


fig,ax=plt.subplots()
ax.plot(yt,label='Train Data')
ax.plot(yf,label='Test Data')
ax.plot(ses_holtf,label='Simple Exponential Smoothing Forecast')
ax.set_title('SES method forecast')
ax.set_xlabel('time')
ax.set_ylabel('data')
ax.legend()
plt.show()

ses_forecast_mse=MSE(yf,ses_holtf.values)
print(f'MSE of SES method :{ses_forecast_mse}')
ses_var_pred_error=var_error(yt,ses_predictions.values)
print(f'variance of prediction error of SES method:{ses_var_pred_error}')
ses_var_forecast_error=var_error(yf,ses_holtf.values)
print(f'variance of forecast error of SES method:{ses_var_forecast_error}')

error_ses=yf-ses_holtf.values
acf_ses = auto_correlation_calculator(error_ses, lags)

plt.stem(np.arange(-lags, lags + 1), np.hstack(((acf_ses[::-1])[:-1],acf_ses)), linefmt='grey', markerfmt='o')
# plt.stem(np.arange(1, lags + 1), acf_ses, linefmt='grey', markerfmt='o')
plt.title("ACF Plot of SES Forecast Errors")
plt.xlabel("Lags")
plt.ylabel("ACF")
plt.grid()
plt.legend(["ACF"], loc='lower right')
plt.show()


ses_qvalue=Qvalue(ses_holtf,lags)
print(f'qvalue of forecast of SES method:{ses_qvalue}')
ses_corr=correlation_coefficent_cal(error_ses,yf)
print(f'correlation coefficient of forecast error and test set of SES method:{ses_corr}')

# Holt linear method

print("---------------Holt linear method----------------")
holtlineart=ets.ExponentialSmoothing(yt,trend='add',damped_trend=False,seasonal=None).fit()
holtlinearf=holtlineart.forecast(steps=len(yf))
holtlinearf=pd.DataFrame(holtlinearf).set_index(yf.index)[0]

holt_linear_prediction = ets.ExponentialSmoothing(yt, trend='add', damped_trend=True, seasonal=None).fit()
holt_linear_prediction_model = holt_linear_prediction.forecast(steps=len(yt))
holt_linear_predictions = pd.DataFrame(holt_linear_prediction_model).set_index(yt.index)[0]

fig,ax=plt.subplots()
ax.plot(yt,label='Train Data')
ax.plot(yf,label='Test Data')
ax.plot(holtlinearf,label='Holt Linear Forecast')
ax.set_title('Holt X    Linear method forecast')
ax.set_xlabel('time')
ax.set_ylabel('data')
ax.legend()
plt.show()

holtlinear_forecast_mse=MSE(yf,holtlinearf.values)
print(f'MSE of holt linear method :{holtlinear_forecast_mse}')
holtlinear_var_pred_error=var_error(yt,holt_linear_predictions.values)
print(f'variance of prediction error of holt linear method:{holtlinear_var_pred_error}')
holtlinear_var_forecast_error=var_error(yf,holtlinearf.values)
print(f'variance of forecast error of holt linear method:{holtlinear_var_forecast_error}')

error_holtlinear=yf-holtlinearf.values
acf_holtlinear = auto_correlation_calculator(error_holtlinear, lags)

plt.stem(np.arange(-lags, lags + 1), np.hstack(((acf_holtlinear[::-1])[:-1],acf_holtlinear)), linefmt='grey', markerfmt='o')
# plt.stem(np.arange(1, lags + 1), acf_holtlinear, linefmt='grey', markerfmt='o')
plt.title("ACF Plot of Holt Linear Forecast Errors")
plt.xlabel("Lags")
plt.ylabel("ACF")
plt.grid()
plt.legend(["ACF"], loc='lower right')
plt.show()


holtlinear_qvalue=Qvalue(holtlinearf,lags)
print(f'qvalue of forecast of Holt linear method:{holtlinear_qvalue}')

holtlinear_corr=correlation_coefficent_cal(error_holtlinear,yf)
print(f'correlation coefficient of forecast error and test set of holt linear method:{holtlinear_corr}')



# Holt Winter method

print("---------------Holt winter method----------------")

holt_wintert=ets.ExponentialSmoothing(yt,trend='mul',damped_trend=True,seasonal='mul',seasonal_periods=24).fit()
holt_winterf=holt_wintert.forecast(steps=len(yf))
holt_winterf=pd.DataFrame(holt_winterf).set_index(yf.index)[0]

holt_winter_prediction = ets.ExponentialSmoothing(yt, trend='add', damped_trend=True, seasonal='mul',seasonal_periods=24).fit()
holt_winter_prediction_model = holt_winter_prediction.forecast(steps=len(yt))
holt_winter_predictions = pd.DataFrame(holt_winter_prediction_model).set_index(yt.index)[0]

fig,ax=plt.subplots()
ax.plot(yt,label='Train Data')
ax.plot(yf,label='Test Data')
ax.plot(holt_winterf,label='Holt Winter Forecast')
ax.set_title('Holt Winter method forecast')
ax.set_xlabel('time')
ax.set_ylabel('data')
ax.legend()
plt.show()

holtwinter_forecast_mse=MSE(yf,holt_winterf.values)
print(f'MSE of holt winter method :{holtwinter_forecast_mse}')
holtwinter_var_pred_error=var_error(yt,holt_winter_predictions.values)
print(f'variance of prediction error of holt winter method:{holtwinter_var_pred_error}')
holtwinter_var_forecast_error=var_error(yf,holt_winterf.values)
print(f'variance of forecast error of holt winter method:{holtwinter_var_forecast_error}')

error_holtwinter=yf-holt_winterf.values
acf_holtwinter = auto_correlation_calculator(error_holtwinter, lags)

plt.stem(np.arange(-lags, lags + 1), np.hstack(((acf_holtwinter[::-1])[:-1],acf_holtwinter)), linefmt='grey', markerfmt='o')
# plt.stem(np.arange(1, lags + 1), acf_holtwinter, linefmt='grey', markerfmt='o')
plt.title("ACF Plot of Holt Winter Forecast Errors")
plt.xlabel("Lags")
plt.ylabel("ACF")
plt.grid()
plt.legend(["ACF"], loc='lower right')
plt.show()


holtwinter_qvalue=Qvalue(holt_winterf,lags)
print(f'qvalue of forecast of Holt winter method:{holtwinter_qvalue}')

holtwinter_corr=correlation_coefficent_cal(error_holtwinter,yf)
print(f'correlation coefficient of forecast error and test set of holt winter method:{holtwinter_corr}')

print("-------------RESULT ---------------------")
result_table = pd.DataFrame(
    columns=["Method", "Forecast_MSE", "Prediction_Var", "Forecast_Var", "Qvalue", "Corr"])
result_table["Method"] = ["Average", "Naive", "Drift", "SES", "Holt Linear", "Holt-Winter"]
result_table["Forecast_MSE"] = [average_forecast_mse,
                          naive_forecast_mse,
                          drift_forecast_mse,
                          ses_forecast_mse,
                          holtlinear_forecast_mse,
                          holtwinter_forecast_mse]

result_table["Prediction_Var"] = [average_var_pred_error,
                            naive_var_pred_error,
                            drift_var_pred_error,
                            ses_var_pred_error,
                            holtlinear_var_pred_error,
                            holtwinter_var_pred_error]

result_table["Forecast_Var"] = [average_var_forecast_error,
                          naive_var_forecast_error,
                          drift_var_forecast_error,
                          ses_var_forecast_error,
                          holtlinear_var_forecast_error,
                          holtwinter_var_forecast_error]
result_table["Qvalue"] = [average_qvalue,
                    naive_qvalue,
                    drift_qvalue,
                    ses_qvalue,
                    holtlinear_qvalue,
                    holtwinter_qvalue]

result_table["Corr"] = [average_corr,
                  naive_corr,
                  drift_corr,
                  ses_corr,
                  holtlinear_corr,
                  holtwinter_corr]
print(result_table)

#multiple linear regression and Feature selection:

print(df.head(5))
# split the dataset into test and train
df
columns_drop=df[['Symbol','Date','Close','Day','Month','Year','Hour','Minute','Second','volatility','Close_LogT','Differential_close','Cumulative Return']]
x= df.drop(columns_drop,axis=1)
print(x)
y=df['Close']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=100)

# 2: plotting the correlation
plt.figure(figsize=(26, 16))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap of ETH.1h dataset', fontdict={'fontsize':20}, pad=12)
plt.tight_layout()
plt.show()

# 3: Collinearity detection:
# a.) Perform SVD analysis on the original feature space
C=np.dot(x.T,x)
U,S,VH=np.linalg.svd(C)
print('all singular values will be: ',list(S))
# We can see that two singular values are close to 0, indicating that there is co-linearity between two or more features.

# b.)Calculate the condition number and write down your observation if co-linearity exists. Justify your answer.
condition_number_cal= np.linalg.cond(x)
print('Condition number: ', condition_number_cal)

# We can see that the condition number is larger than 1000, indicating that the degree of co-linearity is severe.
# c.) # C.) If collinearity exist, how many features will you remove to avoid the co-linearity
# The number of features to be deleted is determined by doing a backward/forward selection process and evaluating the t-test p values,
# then eliminating features one at a time until the modified R square, AIC, and BIC values show a significant change.

#4: use the x-train and y-train dataset and estimate the regression model unknown coefficients using the Normal equation
print('Unknown coefficients check: ',list(LSE(x_train,y_train)))
# #question 5: statsmodels package and OLS function to find the unknown coefficients
#
modeling = sm.OLS(y_train,x_train).fit()
print(modeling.summary())

# We may state that the unknown coefficient results determined using steps 4 and 5 are same
# because the unknown coefficient results are identical.
# We should remove Volume from the equation.


# 6 Feature selection: Using a backward stepwise regression, reduce the feature space dimension.
# use of AIC, BIC and Adjusted R2 as a predictive accuracy for your analysis
#
# dropping Volume from dataset

x_train1=x_train.drop(['Volume'],axis=1)
model0=sm.OLS(y_train,x_train1).fit()
print(model0.summary())

# dropping Market_cap from dataset

x_train1=x_train.drop(['Volume','Market_cap'],axis=1)
model1=sm.OLS(y_train,x_train1).fit()
print(model1.summary())

# dropping Open from dataset

x_train1=x_train.drop(['Volume','Market_cap','Open'],axis=1)
model2=sm.OLS(y_train,x_train1).fit()
print(model2.summary())
# # Because the r-squared, AIC, and BIC values change a lot when we tweak them, we can keep the stroke and call it a day.
#
#7: the reduced feature spaced and display the final model.
# # Final Model is without Volume the Market_cap
#
# print('<========> FINAL MODEL <=========>')
x_train1=x_train.drop(['Volume','Market_cap'],axis=1)
model1=sm.OLS(y_train,x_train1).fit()
print(model1.summary())
#
#8: a prediction for the length of test-set and plot the train, test, and predicted values in one graph.
column_to_drop = x_test[['Volume','Market_cap']]
x_test1=x_test.drop(column_to_drop, axis= 1)
#prediction
y_prediction = model1.predict(x_test1)

plt.figure(figsize = (10, 10))
plt.plot(y_train,label = 'y_train')
plt.plot(y_test,label = 'y_test')
plt.plot(y_prediction,label = 'y_prediction')
plt.title('Prediction of ETH1 dataset',fontsize='20')
plt.xlabel('observation',)
plt.ylabel('Close price in USD',fontsize='20')
plt.grid()
plt.legend()
plt.show()

# 9 the prediction errors and plot the ACF of prediction errors.
y_predictor = model1.predict(x_train1)
y_prediction_error = y_train-y_predictor
lags=20
ACF1 = auto_correlation_calculator(y_prediction_error, lags)
x=np.arange(-lags, lags + 1)
y=np.hstack(((ACF1[::-1])[:-1],ACF1))
plt.stem(x,y, linefmt='grey', markerfmt='o')
plt.title("ACF Plot prediction Errors",fontsize='20')
plt.xlabel("Lags",fontsize='20')
plt.ylabel("ACF",fontsize='20')
plt.grid()
plt.legend(["ACF"], loc='upper right')
plt.show()

# # We can see that the auto correlation is insignificant after  1 delay.
#
#10  the forecast errors and plot the ACF of forecast errors.
forecast_error=y_test-y_prediction
lags=20
ACF2 = auto_correlation_calculator(forecast_error, lags)
x=np.arange(-lags, lags + 1)
y=np.hstack(((ACF2[::-1])[:-1],ACF2))
plt.stem(x,y , linefmt='grey', markerfmt='o')
plt.title("ACF Plot of forecast Errors",fontsize='20')
plt.xlabel("Lags",fontsize='20')
plt.ylabel("ACF",fontsize='20')
plt.grid()
plt.legend(["ACF"], loc='upper right')
plt.show()

# # We can see that the auto correlation is insignificant after only 1 lag.


# 11: the estimated variance of the prediction errors and the forecast errors.
#prediction error
prediction_error=estimated_variance(y_prediction_error,len(x_train1),len(x_train1.columns))
print('estimated variance of prediction error :',prediction_error)

#forecast error
forecast_error_Check=estimated_variance(forecast_error,len(x_train1),len(x_train1.columns))
print('estimated variance of forecast error :',forecast_error_Check)

# We can see that the expected variance of prediction and forecast errors are rather similar.

#12
r1 = np.identity(len(model1.params))
t_test1=model1.t_test(r1)
print('t test final model:', t_test1)
print('p-value of TTest:',t_test1.pvalue)

# because the p-value is less than 5%, reject the null hypothesis # because the regression coefficients are not equal to 0

f_test=model1.f_test(r1)
print('f test of final model:',f_test)
print('p-value of Ftest:',f_test.pvalue)

# we can observe that p-value is smaller that 5 percent , thus we reject the null hypothesis\s# and conclude that your model gives a better fit than the intercept-only model
yt1,yf1 = train_test_split(df.Differential_close,shuffle=False,test_size=0.2)
dtrain,dtest = train_test_split(df.Date,shuffle=False,test_size=0.2)
#  ================================= ARMA Process  =================================

print('========================= ARMA (1,1) model =================================')

acf_y2 = auto_correlation_calculator(df.Differential_close,100)
plot_gpac(calculate_gpac(acf_y2,7,7), "GPAC ARMA (1,1)")


na = 1
nb = 1

model = sm.tsa.ARIMA(yt1, order=(1,0,1)).fit()
print(model.summary())
for i in range(na):
    print(f'The AR Coefficient : a{i+1} is:, {model.params[i]}')

for i in range(nb):
    print(f'The MA Coefficient : b{i+1} is:, {model.params[i+na]}')

# =========== 1 step prediction ================
# def inverse_diff(y20,z_hat,interval=1):
#     y_new = np.zeros(len(y20))
#     for i in range(1,len(z_hat)):
#         y_new[i] = z_hat[i-interval] + y20[i-interval]
#     y_new = y_new[1:]
#     return y_new

def inverse_log_diff(y20,z_hat,interval=1):
    y_new = np.zeros(len(y20))
    for i in range(1,len(z_hat)):
        y_new[i] = np.exp(z_hat[i-interval] + y20[i-interval])
    y_new = y_new[1:]
    return y_new

model_hat = model.predict(start=0,end=len(yt1)-1)

# res_arma_error = np.array(yt) - np.array(model_hat)

y_hat = inverse_log_diff(df.Differential_close[:len(yt1)].values,np.array(model_hat),1)

res_arma_error = df.Differential_close[:len(yt1)-1] - y_hat

lags=100


plot_acf(res_arma_error,lags=lags)
plt.title('ACF of the residual error (ARMA(1,1))')
plt.legend()
plt.show()
plot_pacf(res_arma_error,lags=lags)
plt.title('PACF of the residual error (ARMA(1,1))')
plt.legend()
plt.show()
acf_res = acf(res_arma_error, nlags= lags)
plt.stem(np.arange(-lags, lags+1 ), np.hstack(((acf_res[::-1])[:-1],acf_res)), linefmt='grey', markerfmt='o')
m = 1.96 / np.sqrt(100)
plt.axhspan(-m, m, alpha=.2, color='blue')
plt.title("ACF Plot of residual error (ARMA(1,1))")
plt.xlabel("Lags")
plt.ylabel("ACF values")
plt.grid()
plt.legend(["ACF"], loc='upper right')
plt.show()


plt.plot(df.Date[:99],df.Differential_close[:99], label = 'train set')
plt.plot(df.Date[:99],y_hat[:99], label = '1-step prediction')
plt.title('Close vs time(ARMA(1,1)) - prediction plot')
plt.xlabel('time')
plt.ylabel('Close')
plt.legend()
plt.tight_layout()
plt.show()

# diagnostic testing
from scipy.stats import chi2

print('confidence intervals of estimated parameters:',model.conf_int())

poles = []
for i in range(na):
    poles.append(-(model.params[i]))

print('zero/cancellation:')
zeros = []
for i in range(nb):
    zeros.append(-(model.params[i+na]))

print(f'zeros : {zeros}')
print(f'poles : {poles}')

Q = len(yt)*np.sum(np.square(acf_res[lags:]))

DOF = lags-na-nb

alfa = 0.01

chi_critical = chi2.ppf(1-alfa, DOF)

print('Chi Squared test results')

if Q<chi_critical:
    print(f'The residuals is white, chi squared value :{Q}')
else:
    print(f'The residual is NOT white, chi squared value :{Q}')

# =========== h step prediction ================
forecast = model.forecast(steps=len(yf1))

y_arma_pred = pd.Series(forecast.iloc[0], index=yf1.index)

y_hat_fore = inverse_log_diff(df.Differential_close[len(yt1):].values,np.array(y_arma_pred),1)

# h step prediction

plt.plot(dtest[1:],df.Differential_close[(len(yt1)+1):].values.flatten(), label='Test Data')
plt.plot(dtest[1:],y_hat_fore, label='ARMA Method Forecast')
plt.title('CLose vs time(ARMA(1,1)) - prediction plot')
plt.xlabel('time')
plt.ylabel('Close')
plt.legend()
plt.show()

res_arma_forecast = df.Differential_close[len(yt1)+1:] - y_hat_fore

print(f'variance of residual error : {np.var(res_arma_error)}')

print(f'variance of forecast error : {np.var(res_arma_forecast)}')

# ========================== 2nd ARMA model na=4, nb=4 ===========================
print('========================= ARMA (2,2) model =================================')

na = 2
nb = 2

model1 = sm.tsa.ARIMA(yt1, order=(2,0,2)).fit()
print(model1.summary())

for i in range(na):
    print(f'The AR Coefficient : a{i+1} is:, {model1.params[i]}')

for i in range(nb):
    print(f'The MA Coefficient : b{i+1} is:, {model1.params[i+na]}')


#================= 1 step prediction =====================
model_hat1 = model1.predict(start=0,end=len(yt1)-1)

# res_arma_error = np.array(yt) - np.array(model_hat)

y_hat1 = inverse_log_diff(df.Differential_close[:len(yt1)].values,np.array(model_hat1),1)

res_arma_error1 = df.Differential_close[:len(yt1)-1] - y_hat1

lags=100

plot_acf(res_arma_error,lags=lags)
plt.title('ACF of the residual error (ARMA(2,2))')
plt.legend()
plt.show()
plot_pacf(res_arma_error,lags=lags)
plt.title('PACF of the residual error (ARMA(2,2))')
plt.legend()
plt.show()

acf_res = acf(res_arma_error1, nlags= lags)
plt.stem(np.arange(-lags, lags+1 ), np.hstack(((acf_res[::-1])[:-1],acf_res)), linefmt='grey', markerfmt='o')
m = 1.96 / np.sqrt(100)
plt.axhspan(-m, m, alpha=.2, color='blue')
plt.title("ACF Plot of residual error (ARMA(2,2))")
plt.xlabel("Lags")
plt.ylabel("ACF values")
plt.grid()
plt.legend(["ACF"], loc='upper right')
plt.show()


plt.plot(df.Date[:99],df.Differential_close[:99], label = 'train set')
plt.plot(df.Date[:99],y_hat1[:99], label = '1-step prediction')
plt.title('Close vs time(ARMA(2,2)) - prediction plot')
plt.xlabel('time')
plt.ylabel('Close')
plt.legend()
plt.tight_layout()
plt.show()

# diagnostic testing
from scipy.stats import chi2

print('confidence intervals of estimated parameters:',model1.conf_int())

poles = []
for i in range(na):
    poles.append(-(model1.params[i]))

print('zero/cancellation:')
zeros = []
for i in range(nb):
    zeros.append(-(model1.params[i+na]))

print(f'zeros : {zeros}')
print(f'poles : {poles}')

Q = len(yt1)*np.sum(np.square(acf_res[lags:]))

DOF = lags-na-nb

alfa = 0.01

chi_critical = chi2.ppf(1-alfa, DOF)

print('Chi Squared test results')

if Q<chi_critical:
    print(f'The residuals is white, chi squared value :{Q/100}')
else:
    print(f'The residual is NOT white, chi squared value :{Q/100}')

# =========== h step prediction ================

forecast1 = model1.forecast(steps=len(yf1))

y_arma_pred1 = pd.Series(forecast1.iloc[0], index=yf1.index)

y_hat_fore1 = inverse_diff(y[len(yt1):].values,np.array(y_arma_pred1),1)

# h step prediction

plt.plot(dtest[1:],df.Differential_close[len(yt1)+1:].values.flatten(), label='Test Data')
plt.plot(dtest[1:],y_hat_fore1, label='ARMA Method Forecast')
plt.title('Close vs time(ARMA(2,2)) - prediction plot')
plt.xlabel('time')
plt.ylabel('CLose')
plt.legend()
plt.show()

res_arma_forecast1 = df.Differential_close[len(yt1)+1:] - y_hat_fore1

print(f'variance of residual error : {np.var(res_arma_error1)}')

print(f'variance of forecast error : {np.var(res_arma_forecast1)}')
#================ARIMA order (3,1,18)========================================================

df_train, df_test = train_test_split(df.Close, shuffle=False, test_size=0.3)

print('train shape :', df_train.shape)
print('validation shape :', df_test.shape)
Rolling_average = df["Close_LogT"].rolling(window = 7, center= False).mean()
log_Rolling_difference= df["Close_LogT"] - Rolling_average
log_Rolling_difference.dropna(axis=0,inplace=True)
from statsmodels.tsa.stattools import acf, pacf

#ACF and PACF plots:
lag_acf = acf(log_Rolling_difference, nlags=100)
lag_pacf = pacf(log_Rolling_difference, nlags=100, method='ols')
log_Rolling_difference = log_Rolling_difference.fillna(0)
model = sm.tsa.ARIMA(df["Close_LogT"], order=(3, 1, 0))
results_AR = model.fit()
plt.plot(log_Rolling_difference, label ='Close')
plt.plot(results_AR.fittedvalues, color='red', label = 'order 3')
RSS = results_AR.fittedvalues-log_Rolling_difference
RSS.dropna(inplace=True)
plt.title('RSS: %.4f'% sum(RSS**2))
plt.legend(loc = 'best')
#MA model
model = sm.tsa.ARIMA(df["Close_LogT"], order=(0, 1, 18))
results_MA = model.fit()
plt.plot(log_Rolling_difference)
plt.plot(results_MA.fittedvalues, color='red')
RSS = results_MA.fittedvalues-log_Rolling_difference
RSS.dropna(inplace=True)
plt.title('RSS: %.4f'% sum(RSS**2))
print(results_MA.summary())
plt.plot(df["Close_LogT"], label = 'log_tranfromed_data')
plt.plot(results_MA.resid, color ='green',label= 'Residuals')
plt.title('MA Model Residual plot')
plt.legend(loc = 'best')
results_MA.resid.plot(kind='kde')
plt.title('Density plot of the residual error values')
print(results_MA.resid.describe())

#ARIMA Combined model
model = sm.tsa.ARIMA(df["Close_LogT"], order=(3, 1, 18))
results_ARIMA = model.fit()
plt.plot(log_Rolling_difference)
plt.plot(results_ARIMA.fittedvalues, color='red', label = 'p =8, q =18')
RSS =results_ARIMA.fittedvalues-log_Rolling_difference
RSS.dropna(inplace=True)
plt.title('RSS: %.4f'% sum(RSS**2))
plt.legend(loc='best')
print(results_ARIMA.summary())
plt.plot(df["Close_LogT"], label = 'log_tranfromed_data')
plt.plot(results_ARIMA.resid, color ='green',label= 'Residuals')
plt.title('ARIMA Model Residual plot')
plt.legend(loc = 'best')

results_ARIMA.resid.plot(kind='kde')
plt.title('Density plot of the residual error values')
print(results_ARIMA.resid.describe())

train, test = train_test_split(df,shuffle=False,test_size=0.3)
Test_y=test.Close
test.drop(columns='Close', inplace=True)

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())
predictions_ARIMA_log = pd.Series(df.Close_LogT.iloc[0], index=df.Close_LogT.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(df.Close,label='Actual Close')
plt.plot(predictions_ARIMA, label='Predicted Close')
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-df.Close)**2)/len(df.Close)))
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
dates=test.index
forecast = pd.Series(results_ARIMA.forecast(steps=10350).iloc[0],dates)
forecast = 10**(forecast+test.Differential_close)
print(forecast)
error = mean_squared_error(Test_y, forecast)
print('Test MSE: %.3f' % error)


#==============================Arima order(2,2,10)==============

#ARIMA Combined model
model = sm.tsa.ARIMA(df["Close_LogT"], order=(2, 2, 0))
results_AR = model.fit()
plt.plot(log_Rolling_difference, label ='Close')
plt.plot(results_AR.fittedvalues, color='red', label = 'order 3')
RSS = results_AR.fittedvalues-log_Rolling_difference
RSS.dropna(inplace=True)
plt.title('RSS: %.4f'% sum(RSS**2))
plt.legend(loc = 'best')
#MA model
model = sm.tsa.ARIMA(df["Close_LogT"], order=(0, 2, 10))
results_MA = model.fit()
plt.plot(log_Rolling_difference)
plt.plot(results_MA.fittedvalues, color='red')
RSS = results_MA.fittedvalues-log_Rolling_difference
RSS.dropna(inplace=True)
plt.title('RSS: %.4f'% sum(RSS**2))
print(results_MA.summary())
plt.plot(df["Close_LogT"], label = 'log_tranfromed_data')
plt.plot(results_MA.resid, color ='green',label= 'Residuals')
plt.title('MA Model Residual plot')
plt.legend(loc = 'best')
results_MA.resid.plot(kind='kde')
plt.title('Density plot of the residual error values')
print(results_MA.resid.describe())

#ARIMA Combined model
model = sm.tsa.ARIMA(df["Close_LogT"], order=(2, 2, 10))
results_ARIMA = model.fit()
plt.plot(log_Rolling_difference)
plt.plot(results_ARIMA.fittedvalues, color='red', label = 'p =8, q =18')
RSS =results_ARIMA.fittedvalues-log_Rolling_difference
RSS.dropna(inplace=True)
plt.title('RSS: %.4f'% sum(RSS**2))
plt.legend(loc='best')
print(results_ARIMA.summary())
plt.plot(df["Close_LogT"], label = 'log_tranfromed_data')
plt.plot(results_ARIMA.resid, color ='green',label= 'Residuals')
plt.title('ARIMA Model Residual plot')
plt.legend(loc = 'best')

results_ARIMA.resid.plot(kind='kde')
plt.title('Density plot of the residual error values')
print(results_ARIMA.resid.describe())

train, test = train_test_split(df,shuffle=False,test_size=0.3)
Test_y=test.Close
test.drop(columns='Close', inplace=True)

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())
predictions_ARIMA_log = pd.Series(df.Close_LogT.iloc[0], index=df.Close_LogT.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(df.Close,label='Actual Close')
plt.plot(predictions_ARIMA, label='Predicted Close')
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-df.Close)**2)/len(df.Close)))
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
dates=test.index
forecast = pd.Series(results_ARIMA.forecast(steps=10350).iloc[0],dates)
forecast = 10**(forecast+test.Differential_close)
print(forecast)
error = mean_squared_error(Test_y, forecast)
print('Test MSE: %.3f' % error)









