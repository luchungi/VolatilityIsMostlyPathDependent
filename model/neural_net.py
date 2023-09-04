import numpy as np
import pandas as pd
import torch # faster with torch than numpy

def exp_kernel(t, λ, θ):
    '''
    Exponential kernel function torch version
    params:
    t: torch.tensor
        time difference between prediction datetime and each row in the sliding window and is a tensor of shape (len(df) - n, n)
    λ: float
        decay rate of the exponential kernel function
    θ: float
        weight of the exponential kernel function
    '''
    return θ * λ * torch.exp(-λ * t)

def convex_combi_exp_kernel(t, λ1, λ2, θ):
    '''
    Convex combination of two exponential kernel functions torch version
    params:
    t: torch.tensor
        time difference between prediction datetime and each row in the sliding window and is a tensor of shape (len(df) - n, n)
    λ1: float
        decay rate of the first exponential kernel function
    λ2: float
        decay rate of the second exponential kernel function
    θ: float
        weight of the second exponential kernel function and the weight of the first exponential kernel function is 1 - θ
    '''
    return exp_kernel(t, λ1, 1-θ) + exp_kernel(t, λ2, θ)

def R1(λ1, λ2, θ, return_tensor, t):
    '''
    Calculate R1 which is the simple return weighted by convex combination of two exponential kernel functions torch version
    params:
    λ1: float
        decay rate of the first exponential kernel function
    λ2: float
        decay rate of the second exponential kernel function
    θ: float
        weight of the second exponential kernel function and the weight of the first exponential kernel function is 1 - θ
    return_tensor: torch.tensor
        simple return of the sliding window and is a tensor of shape (len(df) - n, n)
    t: torch.tensor
        time difference between prediction datetime and each row in the sliding window and is a tensor of shape (len(df) - n, n)
    '''

    return torch.sum(return_tensor * convex_combi_exp_kernel(t, λ1, λ2, θ), dim=1)

def R2(λ1, λ2, θ, return_tensor, t):
    '''
    Calculate R2 which is the squared return weighted by convex combination of two exponential kernel functions torch version
    params:
    λ1: float
        decay rate of the first exponential kernel function
    λ2: float
        decay rate of the second exponential kernel function
    θ: float
        weight of the second exponential kernel function and the weight of the first exponential kernel function is 1 - θ
    return_tensor: torch.tensor
        simple return of the sliding window and is a tensor of shape (len(df) - n, n)
    t: torch.tensor
        time difference between prediction datetime and each row in the sliding window and is a tensor of shape (len(df) - n, n)
    '''
    return torch.sqrt(torch.sum(return_tensor * convex_combi_exp_kernel(t, λ1, λ2, θ), dim=1))

def predict(β0, β1, β2, λ11, λ12, θ1, λ21, λ22, θ2, df, n):
    '''
    Predict the VIX level using the PDV model torch version
    params:
    β0: float
        intercept of the PDV model
    β1: float
        coefficient of R1
    β2: float
        coefficient of R2
    λ11: float
        decay rate of the first exponential kernel function of R1
    λ12: float
        decay rate of the second exponential kernel function of R1
    θ1: float
        weight of the second exponential kernel function of R1 and the weight of the first exponential kernel function is 1 - θ1
    λ21: float
        decay rate of the first exponential kernel function of R2
    λ22: float
        decay rate of the second exponential kernel function of R2
    θ2: float
        weight of the second exponential kernel function of R2 and the weight of the first exponential kernel function is 1 - θ2
    df: pandas.DataFrame
        dataframe containing the data with columns 'r1', 'r2', 'vix' where 'r1' is the simple return, 'r2' is the squared return and 'vix' is the VIX level to be predicted
    n: int
        sliding window size to calculate R1 and R2
    '''
    array = df[['r1', 'r2', 'vix']].values

    # create sliding window of size n for r1 and r2 with shape (len(df) - n + 1, n)
    r1_sliding_window = torch.tensor(np.lib.stride_tricks.sliding_window_view(array[:, 0], n), dtype=torch.float64, requires_grad=False)
    r2_sliding_window = torch.tensor(np.lib.stride_tricks.sliding_window_view(array[:, 1], n), dtype=torch.float64, requires_grad=False)

    # calculate time difference between prediction datetime and each row in the sliding window with shape (len(df) - n + 1, n)
    dt = ((df.index[1:] - df.index[:-1]).days / 365).values
    dt_sliding_window = np.lib.stride_tricks.sliding_window_view(dt, n-1)
    dt_sliding_window = np.flip(dt_sliding_window, axis=1)
    dt_sliding_window = np.cumsum(dt_sliding_window, axis=1)
    dt_sliding_window = np.flip(dt_sliding_window, axis=1)
    dt_sliding_window = np.concatenate((dt_sliding_window, np.zeros((dt_sliding_window.shape[0], 1))), axis=1)
    t = torch.tensor(dt_sliding_window, dtype=torch.float64, requires_grad=False)

    return β0 + β1 * R1(λ11, λ12, θ1, r1_sliding_window, t) + β2 * R2(λ21, λ22, θ2, r2_sliding_window, t)

def residual(x, df, n):
    '''
    Calculate the residuals of the PDV model torch version
    params:
    x: list
        list of parameters of the PDV model
    df: pandas.DataFrame
        dataframe containing the data with columns 'r1', 'r2', 'vix' where 'r1' is the simple return, 'r2' is the squared return and 'vix' is the VIX level to be predicted
    n: int
        sliding window size to calculate R1 and R2
    '''
    β0, β1, β2, λ11, λ12, θ1, λ21, λ22, θ2 = x
    y = df.iloc[n-1:]['vix'].values # shape (len(df) - n + 1,) = number of predictions
    y_hat = predict(β0, β1, β2, λ11, λ12, θ1, λ21, λ22, θ2, df, n)
    return y - y_hat.numpy()

def create_dataset_from_yf_df(spx_df, vix_df, start_date, end_date, predict_t_plus_1=False):
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
    '''
    # check that both dataframe have a datetime index
    if not isinstance(spx_df.index, pd.DatetimeIndex):
        raise ValueError('spx_df must have a datetime index')
    if not isinstance(vix_df.index, pd.DatetimeIndex):
        raise ValueError('vix_df must have a datetime index')

    # calculate simple return and squared return of S&P500
    spx = pd.DataFrame(columns=['r1', 'r2'])
    spx['r1'] = spx_df.loc[start_date:end_date-pd.Timedelta(days=1), 'Close'].pct_change()
    spx['r2'] = spx['r1'] ** 2

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
