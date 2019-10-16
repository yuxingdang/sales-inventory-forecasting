#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:44:01 2019

@author: yudi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df= pd.read_csv('train.csv',parse_dates=['date'],index_col=['date'])

#check data & preprocessing
df.head()
df.info() #df.shape df.dtypes
np.unique(df.store)
np.unique(df.item)

df['sales'].astype(float)

def sales_store(i):
    sales_store=df[df.store==i]['sales'].sort_index(ascending=True)
    return sales_store

plt.figure()
c = '#386B7F'
for i in range(1,11):
    plt.subplot(3,4,i)
    sales_store(i).resample('w').sum().plot(color=c)

decomposition_1=sm.tsa.seasonal_decompose(sales_store(1),model='addictive',freq=365)
decomposition_1.trend.plot()

#time series decomposition
sales = df.drop(['store','item'], axis=1).copy() #temporary array
sales_monthly= sales['sales'].resample('MS').sum()
sales_monthly.plot()

decomposition = sm.tsa.seasonal_decompose(sales_monthly, model='additive')
decomposition.plot()

decomposition = sm.tsa.seasonal_decompose(sales_monthly, model='multiplicative')
decomposition.plot()

#modelling-build ARIMA
forecast_add = pd.DataFrame({'month':[],'sales_forecast':[],'item':[],'store':[]})

import itertools

#parameter range
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(i[0], i[1], i[2], 12) for i in pdq]

#function -- optimize parameter
def param_function(y):   
    count=0
    pdq_test=[]
    seasonal_pdq_test=[]
    AIC=[]
    
    for i in pdq:
        for j in seasonal_pdq:
            try:
                model = sm.tsa.statespace.SARIMAX(y,
                                            order=i,
                                            seasonal_order=j,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
                result = model.fit()
                count += 1
                pdq_test.append(i)
                seasonal_pdq_test.append(j)
                AIC.append(result.aic)
            except:
                continue
    k = AIC.index(min(AIC))
    pdq_opt = pdq_test[k]
    seasonal_pdq_opt = seasonal_pdq_test[k]
    param_opt = [pdq_opt,seasonal_pdq_opt]
    return param_opt

#function -- forecast based on the model
def forecast_function(y):
    param_opt = param_function(y)
    pdq_opt = param_opt[0]
    seasonal_pdq_opt = param_opt[1]

    model = sm.tsa.statespace.SARIMAX(y,
                                order=pdq_opt,
                                seasonal_order=seasonal_pdq_opt,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    result = model.fit()

    #forecast
    try_forecast = result.get_forecast(steps=24)
    forecast = try_forecast.predicted_mean
    
    #type(forecast) change datatype
    dict_forecast = {'month':forecast.index,'sales_forecast':forecast.values}
    df_forecast = pd.DataFrame(dict_forecast)
    
    df_forecast['item'] = m
    df_forecast['store'] = n
    
    global forecast_add
    forecast_add.append(df_forecast,ignore_index=True)

    forecast_add = forecast_add.append(df_forecast,ignore_index=True)
    return forecast_add

#forecast certain store & item
m=5
n=4
df_new = df.query('item == {} & store == {}'.format(m,n))
y = df_new['sales'].resample('MS').sum()
forecast_function(y)

#for m in range(1,51):
#    for n in range(1,11): 
#        df_new = df.query('item == {} & store == {}'.format(m,n))
#        y = df_new['sales'].resample('MS').sum()
#        forecast_function(y)
#print(forecast_add)

#consider feature engineering for date 
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error

def create_features(df, label=None):
    
    #consider feature engineering for date if needed (figure out details)
    df_datefe = df.reset_index(drop=False)
    df_datefe['dayofmonth'] = df_datefe.date.dt.day
    df_datefe['dayofyear'] = df_datefe.date.dt.dayofyear
    df_datefe['dayofweek'] = df_datefe.date.dt.dayofweek
    df_datefe['month'] = df_datefe.date.dt.month
    df_datefe['year'] = df_datefe.date.dt.year
    df_datefe['weekofyear'] = df_datefe.date.dt.weekofyear
    df_datefe['is_month_start'] = (df_datefe.date.dt.is_month_start).astype(int)
    df_datefe['is_month_end'] = (df_datefe.date.dt.is_month_end).astype(int)
    
    X = df_datefe[df_datefe.store==4][df_datefe.item==5][['dayofmonth','dayofyear','dayofweek','month','year',
           'weekofyear','is_month_start','is_month_end']]
    if label:
        y = df_datefe[df_datefe.store==4][df_datefe.item==5]['sales']
        return X, y
    return X

def create_features2(df, label=None):
    
    #consider feature engineering for date if needed (figure out details)
    df_datefe = df.reset_index(drop=True)
    df_datefe['dayofmonth'] = df_datefe.date.dt.day
    df_datefe['dayofyear'] = df_datefe.date.dt.dayofyear
    df_datefe['dayofweek'] = df_datefe.date.dt.dayofweek
    df_datefe['month'] = df_datefe.date.dt.month
    df_datefe['year'] = df_datefe.date.dt.year
    df_datefe['weekofyear'] = df_datefe.date.dt.weekofyear
    df_datefe['is_month_start'] = (df_datefe.date.dt.is_month_start).astype(int)
    df_datefe['is_month_end'] = (df_datefe.date.dt.is_month_end).astype(int)
    
    X = df_datefe[['dayofmonth','dayofyear','dayofweek','month','year',
           'weekofyear','is_month_start','is_month_end']]
    return X

split_date = '2016-12-31'
df_train = df.loc[df.index <= split_date].copy()
df_test = df.loc[df.index > split_date].copy()

X_train,y_train=create_features(df_train,label='sales')
X_test,y_test=create_features(df_test,label='sales')
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
       verbose=False) 

plot_importance(reg, height=0.9)

def generate_time_series(
        start_date,
        cnt, delta, timestamp=False
):
    """
    generate a time series/index
    :param start_date: start date
    :param cnt: date count. If =cnt are specified, delta must not be; one is required
    :param delta: time delta, default is one day.
    :param timestamp: output timestamp or format string
    :return: list of time string or timestamp
    """

    def per_delta():
        curr = start_date + delta
        while curr < end_date:
            yield curr
            curr += delta

    end_date = start_date + delta * cnt

    time_series = []
    if timestamp:
        for t in per_delta():
            time_series.append(t.timestamp())
    else:
        for t in per_delta():
            time_series.append(t)
        # print(t.strftime("%Y-%m-%d"))
    return time_series

generate_time_series(start_date=latest_date, cnt=366, delta=datetime.timedelta(days=1))

latest_date = df_test['date'].tolist()[-1]
forecast_data = pd.DataFrame.from_dict({'date': generate_time_series(start_date=latest_date, cnt=366, delta=datetime.timedelta(days=1))})
forecast_data = forecast_data.set_index(pd.DatetimeIndex(forecast_data['date']))
X_forecast = create_features2(forecast_data,label='sales')

forecast_data['sales_forecast'] = reg.predict(X_forecast)
print(forecast_data['sales_forecast'].resample('MS').sum())