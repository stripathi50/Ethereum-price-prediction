import numpy as np
import matplotlib.pyplot as plt
# from statsmodels.tsa.arima_process import arma_generate_sample
import statsmodels.api as sm
import seaborn as sns
import warnings
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)


def Cal_rolling_mean_var(time,col,ColName1):
    Column1 = pd.Series(col)
    new_mean = []
    new_var = []
    for i in range(Column1.size):
        new_mean.append(np.mean(Column1.head(i)))
        new_var.append(np.var(Column1.head(i)))

    plt.figure(figsize=(15, 8))
    plt.plot(time, new_mean, label=ColName1)
    plt.legend()
    plt.grid()
    plt.xlabel('time',fontsize='20')
    plt.xticks(rotation=90)
    plt.ylabel(f'{ColName1}',fontsize='20')
    plt.title(f'({ColName1} vs time) rolling mean',fontsize='20')
    plt.show()

    plt.figure(figsize=(15, 8))
    plt.plot(time, new_var, label=ColName1)
    plt.legend()
    plt.grid()
    plt.xlabel('time',fontsize='20')
    plt.xticks(rotation=90)
    plt.ylabel(f'{ColName1}',fontsize='20')
    plt.title(f'({ColName1} vs time) rolling variance',fontsize='20')
    plt.show()

def LSE(x,y):
    inverse_mat1=np.linalg.inv(np.dot(x.T,x))
    inverse_mat_mul=np.dot(x.T, y)
    return np.dot(inverse_mat1,inverse_mat_mul)

def get_autocorr(y, lag):
    mean_y = np.mean(y)
    sub_y = y - mean_y
    est_k = sum((sub_y[lag:]) * (sub_y[:-lag]))/sum(sub_y**2)
    return est_k

def plot_autocorr(df, lags):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    autocorr = [1]
    index = [0]
    for k in range(1, lags):
        est_k = get_autocorr(df, k)
        autocorr.append(est_k)
        index.append(k)
        index.insert(0, -k)
        if k > 0:
            autocorr.insert(0, est_k)
    
    
    ax.stem(index, autocorr)
    ax.set_xlabel('lags')
    ax.set_ylabel('Magnitude')
    ax.set_title('Autocorrelation plot')
    plt.show()


def generate_arma(AR_coeff, MA_coeff, num_of_samples, var, mean_e):
    arma_process = sm.tsa.ArmaProcess(AR_coeff, MA_coeff)

    mean_y = mean_e * (1 + sum(MA_coeff)) / (1 + sum(AR_coeff))
    y = arma_process.generate_sample(num_of_samples, scale=var) + mean_y

    return y

def calculate_gpac(ry2, len_ar, len_na):
    
    len_ar +=1
    # len_na +=1
    gpac = np.empty(shape=(len_na, len_ar))
    
    # middle_index = int(len(ry2)/2) 
    
    for k in range(1, len_ar):
            # print("k", k)
            num = np.empty(shape=(k,k))
            denom = np.empty(shape=(k,k))
            for j in range(0, len_na):
                
                for x in range(0,k):
                    for x2 in range(0, k):
                        # all columns expect last
                        if x < k-1:
                            # print("Column one",np.abs(j+x2-x))
                            num[x2][x] = ry2[np.abs(j+x2-x)]
                            denom[x2][x] = ry2[np.abs(j+x2-x)]
                        # last column
                        else:
                            # print("last col",np.abs(j + x2 + 1), ry2[np.abs(j + x2 + 1)])
                            num[x2][x] = ry2[np.abs(j + x2 + 1)]
                            denom[x2][x] = ry2[np.abs(j - k + x2 + 1)]
                       
                            
                num_det = round(np.linalg.det(num),5)
                denom_det = round(np.linalg.det(denom), 5)
                gpac_value = round(num_det/denom_det, 3)
                # print("gpac_value", gpac_value)
                
                if denom_det == 0.0:
                    gpac_value = np.inf
                gpac[j][k] = gpac_value
                    
    gpac = pd.DataFrame(gpac[:,1:])
    gpac.columns = [i for i in range(1,len_ar)]
    
    return gpac

def plot_gpac(gpac, title):
    ax = sns.heatmap(gpac, annot=True)
    plt.title(title)
    plt.show()


def perform_box_pierce_text(train_set_len, train_error):
    correlations = []
    for i in range(1, 20):
        correlations.append(get_autocorr(train_error, i) ** 2)

    Q = train_set_len * sum(correlations)
    return Q

def confidence_interval(theta, cov):
    print("confidence interval of parameter")
    for i in range(len(theta)):
        lb = theta[i] - 2*np.sqrt(cov[i,i])
        ub = theta[i] + 2*np.sqrt(cov[i,i])

        print("{} < theta_{} < {}".format(lb, i, ub))


def calculate_MA(a, m, folding_order, user_input_flag):
    if user_input_flag:
        m = 1
        while m < 3:
            m = int(input("Enter the order of moving average (expect 1,2): "))

        if (m == 1) | (m == 2):
            print("Folding order of m=1,2 will not be accepted")
            return 0

        if m % 2 == 0:
            folding_order = int(input("Please enter even folding order"))

    if m % 2 == 0:
        k = int((m - 1) / 2)
        index = int(m - k - 2)
        ma = np.empty(len(a))
        ma[:] = np.NaN
        for i in range(k, len(a) - index):
            ma[i] = np.mean(a[i - index:i + index + 2])

        folds = 2
        k = int(m / folds)
        # index = int(folds-k-1)
        ma_df = np.empty(len(a))
        ma_df[:] = np.NaN
        for i in range(k, len(a) - index - 1):
            ma_df[i] = np.mean(ma[i - 1:i + 1])

    else:
        k = int((m - 1) / 2)
        index = int(m - k - 1)

        ma_df = np.empty(len(a))
        ma_df[:] = np.NaN

        for i in range(k, len(a) - index):
            ma_df[i] = np.mean(a[i - index:i + index + 1])

    return ma_df

def plot_detrended(data, MA, detrended, xlabel, ylabel, title, serieslabel):
    plt.figure(figsize=[8, 5])
    plt.plot_date(data.index, data, ls='solid', c='green', label='original', marker='')
    plt.plot_date(data.index, MA, ls='solid', c='red', label=serieslabel, marker='')
    plt.plot_date(data.index, detrended, ls='solid', c='orange', label='detrended', marker='')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def get_theoretical_acf(arparams, maparams, lags):
    arma_process = sm.tsa.ArmaProcess(arparams, maparams)

    ry = arma_process.acf(lags=lags)

    ry1 = ry[::-1]
    ry2 = np.concatenate((np.reshape(ry1, lags), ry[1:]))
    return ry2

def estimated_variance(error,T,k):
    import math
    error_square=error**2
    error_square_sum=np.sum(error_square)
    return (math.sqrt(error_square_sum/(T-k-1)))

def inverse_diff(y20,z_hat,interval=1):
    y_new = np.zeros(len(y20))
    for i in range(1,len(z_hat)):
        y_new[i] = z_hat[i-interval] + y20[i-interval]
    y_new = y_new[1:]
    return y_new