{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import datetime as dt\n",
    "from statsmodels.tsa.stattools import acf\n",
    "\n",
    "\n",
    "def read_data()\n",
    "    df = pd.read_csv('./Data/train_fwYjLYX.csv')\n",
    "    testdf = pd.read_csv('./Data/test_1eLl9Yf.csv')\n",
    "    df_1 = df.loc[df['segment']==1]\n",
    "    df_2 = df.loc[df['segment']==2]\n",
    "    return df_1,df_2,testdf\n",
    "\n",
    "def format_ts(df,col,sampling_freq='D'):\n",
    "        df.application_date = pd.to_datetime(df.application_date)\n",
    "        df = df[['application_date',col]]\n",
    "        print(\"Time period given: \",df.application_date.max(), df.application_date.min())\n",
    "        df.columns=['ds','y']\n",
    "        ts1 = df.set_index('ds').resample(sampling_freq).sum()\n",
    "        mindate = str(ts1.index[0])[:10]\n",
    "        if make_test_pred==0:\n",
    "            maxdate = str(ts1.index[-1]-dt.timedelta(days=future_units))[:10]\n",
    "        else:\n",
    "            maxdate = str(ts1.index[len(ts1)-1])[:10]\n",
    "        print(\"Time period used for training: \",mindate, maxdate)\n",
    "        #ts1 = ts1.loc[ts1.index>mindate]\n",
    "        return ts1,mindate,maxdate\n",
    "    \n",
    "# Performance metrics\n",
    "def forecast_accuracy(forecast, actual):\n",
    "    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE\n",
    "    me = np.mean(forecast - actual)             # ME\n",
    "    mae = np.mean(np.abs(forecast - actual))    # MAE\n",
    "    mpe = np.mean((forecast - actual)/actual)   # MPE\n",
    "    rmse = np.mean((forecast - actual)**2)**.5  # RMSE\n",
    "    corr = np.corrcoef(forecast, actual)[0,1]   # corr\n",
    "    mins = np.amin(np.hstack([forecast[:,None], \n",
    "                              actual[:,None]]), axis=1)\n",
    "    maxs = np.amax(np.hstack([forecast[:,None], \n",
    "                              actual[:,None]]), axis=1)\n",
    "    minmax = 1 - np.mean(mins/maxs)             # minmax\n",
    "    acf1 = acf(forecast-actual)[1]                      # ACF1\n",
    "    return({'mape':mape, 'me':me, 'mae': mae, \n",
    "            'mpe': mpe, 'rmse':rmse, 'acf1':acf1,\n",
    "            'corr':corr, 'minmax':minmax})"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "tfgpu",
   "language": "python",
   "name": "tfgpu"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
