import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.tsa.stattools import acf


def read_data():
    df = pd.read_csv('./Data/train_fwYjLYX.csv')
    testdf = pd.read_csv('./Data/test_1eLl9Yf.csv')
    df_1 = df.loc[df['segment']==1]
    df_2 = df.loc[df['segment']==2]
    return df_1,df_2,testdf

def format_ts(df,col,future_units=91,sampling_freq='D',make_test_pred=0):
        df.application_date = pd.to_datetime(df.application_date)
        df = df[['application_date',col]]
        print("Time period given: ",df.application_date.min(),df.application_date.max())
        df.columns=['ds','y']
        ts1 = df.set_index('ds').resample(sampling_freq).sum()
        ts1=ts1.sort_index()
        mindate = str(ts1.index[0])[:10]
        if make_test_pred==0:
            maxdate = str(ts1.index[-1]-dt.timedelta(days=future_units))[:10]
        else:
            maxdate = str(ts1.index[-1])[:10]
        print("Time period used for training: ",mindate, maxdate)
        #ts1 = ts1.loc[ts1.index>mindate]
        return ts1,mindate,maxdate
    
# Performance metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(forecast-actual)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1,
            'corr':corr, 'minmax':minmax})
