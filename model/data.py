import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def create_df_from_yf(spx_df, start_date, end_date, vix_df=None, predict_t_plus_1=False, log_return=False):
    '''
    Create dataset format used for model defined above from yfinance dataframe
    params:
    spx_df: pandas.DataFrame
        dataframe containing the S&P500 data from yfinance with datetime index
    vix_df: pandas.DataFrame
        dataframe containing the VIX data from yfinance with datetime index
    start_date: datetime
        start date of the dataset (included)
    end_date: datetime
        end date of the dataset (excluded)
    predict_t_plus_1: bool
        if True, the model will predict the VIX level at t+1, if False, the model will predict the VIX level at t
    log_return: bool
        if True, the model will use log return, if False, the model will use simple return
    '''
    # check that both dataframe have a datetime index
    if not isinstance(spx_df.index, pd.DatetimeIndex):
        raise ValueError('spx_df must have a datetime index')

    # calculate simple or log return and squared return of S&P500
    spx = pd.DataFrame(columns=['r1', 'r2'])
    if log_return:
        spx['r1'] = np.log(spx_df.loc[start_date:end_date-pd.Timedelta(days=1), 'Close']).diff()
    else:
        spx['r1'] = spx_df.loc[start_date:end_date-pd.Timedelta(days=1), 'Close'].pct_change()
    spx['r2'] = spx['r1'] ** 2

    if vix_df is None:
        combined = spx
    else:
        if not isinstance(vix_df.index, pd.DatetimeIndex):
            raise ValueError('vix_df must have a datetime index')

        # extract VIX level
        vix = vix_df.loc[start_date:end_date-pd.Timedelta(days=1), ['Close']] / 100
        vix.columns = ['vix']

        # check that both dataframes have same index
        if not spx.index.equals(vix.index):
            raise ValueError('spx_df and vix_df must have same index from start_date to end_date')

        # shift VIX level if predict_t_plus_1 is True i.e. predict VIX level at t+1
        if predict_t_plus_1:
            vix = vix.shift(-1)

        combined = pd.concat([spx, vix], axis=1)
        combined = combined.dropna()

    return combined

def batch_create_df_from_yf(data, log_return=False):
    '''
    Create dataset format used for model defined above from yfinance dataframe
    params:
    data: numpy.ndarray
        array of shape (n_samples, n_periods) where each row is a time series of simulated S&P500 price
    vix_df: pandas.DataFrame
        dataframe containing the VIX data from yfinance with datetime index
    start_date: datetime
        start date of the dataset (included)
    end_date: datetime
        end date of the dataset (excluded)
    predict_t_plus_1: bool
        if True, the model will predict the VIX level at t+1, if False, the model will predict the VIX level at t
    '''

    # calculate simple or log return and squared return of S&P500
    features = np.empty((data.shape[0], data.shape[1]-1, 2))

    if log_return:
        features[:, :, 0] = np.log(data[:, 1:]) - np.log(data[:, :-1])
    else:
        features[:, :, 0] = (data[:, 1:] - data[:, :-1]) / data[:, :-1]
    features[:, :, 1] = features[:, :, 0] ** 2

    return features

class VIXDataset(Dataset):
    '''
    Dataset class for VIX data
    x = (r1_sliding_window, r2_sliding_window, t) each with shape (len(df) - n + 1, n)
    y = vix level at last time step of x
    '''
    def __init__(self, df, n, dtype=torch.float32):
        '''
        params:
        df: pandas.DataFrame
            dataframe containing the data with columns 'r1', 'r2', 'vix' where 'r1' is the simple return, 'r2' is the squared return and 'vix' is the VIX level to be predicted
        n: int
            sliding window size to calculate R1 and R2
        '''
        self.len = len(df) - n + 1
         # create sliding window of size n for r1 and r2 with shape (len(df) - n + 1, n)
        r1_sliding_window = np.lib.stride_tricks.sliding_window_view(df.iloc[:, 0].values, n)
        r2_sliding_window = np.lib.stride_tricks.sliding_window_view(df.iloc[:, 1].values, n)

        # calculate time difference between prediction datetime and each row in the sliding window with shape (len(df) - n + 1, n)
        dt = ((df.index[1:] - df.index[:-1]).days / 365).values
        dt_sliding_window = np.lib.stride_tricks.sliding_window_view(dt, n-1)
        dt_sliding_window = np.flip(dt_sliding_window, axis=0)
        dt_sliding_window = np.cumsum(dt_sliding_window, axis=0)
        dt_sliding_window = np.flip(dt_sliding_window, axis=0)
        dt_sliding_window = np.concatenate((dt_sliding_window, np.zeros((dt_sliding_window.shape[0], 1))), axis=1)

        # create sliding window of size n for vix with shape (len(df) - n + 1, n)
        vix_sliding_window = np.lib.stride_tricks.sliding_window_view(df.iloc[:, 2].values, n)

        data = np.stack((r1_sliding_window, r2_sliding_window, dt_sliding_window, vix_sliding_window), axis=1)
        self.data = torch.tensor(data, dtype=dtype, requires_grad=False)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx, :-1, :], self.data[idx, -1, -1]

class VIXDataset2(Dataset):
    '''
    Dataset class for VIX data
    x = (x1, x2) where
        x1 = (r1_sliding_window, r2_sliding_window, t) each with shape (len(df) - n + 1, n)
        x2 = vix from 1st time step to 2nd last time step of x1 with shape (len(df) - n + 1, n-1)
    y = vix level at last time step of x
    '''
    def __init__(self, df, n, dtype=torch.float32):
        '''
        params:
        df: pandas.DataFrame
            dataframe containing the data with columns 'r1', 'r2', 'vix' where 'r1' is the simple return, 'r2' is the squared return and 'vix' is the VIX level to be predicted
        n: int
            sliding window size to calculate R1 and R2
        '''
        self.len = len(df) - n + 1
         # create sliding window of size n for r1 and r2 with shape (len(df) - n + 1, n)
        r1_sliding_window = np.lib.stride_tricks.sliding_window_view(df.iloc[:, 0].values, n)
        r2_sliding_window = np.lib.stride_tricks.sliding_window_view(df.iloc[:, 1].values, n)

        # calculate time difference between prediction datetime and each row in the sliding window with shape (len(df) - n + 1, n)
        dt = ((df.index[1:] - df.index[:-1]).days / 365).values
        dt_sliding_window = np.lib.stride_tricks.sliding_window_view(dt, n-1)
        dt_sliding_window = np.flip(dt_sliding_window, axis=0)
        dt_sliding_window = np.cumsum(dt_sliding_window, axis=0)
        dt_sliding_window = np.flip(dt_sliding_window, axis=0)
        dt_sliding_window = np.concatenate((dt_sliding_window, np.zeros((dt_sliding_window.shape[0], 1))), axis=1)

        # create sliding window of size n for vix with shape (len(df) - n + 1, n)
        vix_sliding_window = np.lib.stride_tricks.sliding_window_view(df.iloc[:, 2].values, n)

        data = np.stack((r1_sliding_window, r2_sliding_window, dt_sliding_window, vix_sliding_window), axis=1)
        self.data = torch.tensor(data, dtype=dtype, requires_grad=False)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return torch.cat([torch.flatten(self.data[idx, :-1, :]), self.data[idx, -1, :-1]]), self.data[idx, -1, -1]