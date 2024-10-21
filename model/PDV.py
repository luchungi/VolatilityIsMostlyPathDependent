import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
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

        return torch.sum(return_tensor * convex_combi_exp_kernel(t, λ1, λ2, θ), dim=-1)

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
    return torch.sqrt(torch.sum(return_tensor * convex_combi_exp_kernel(t, λ1, λ2, θ), dim=-1))

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
    array = df[['r1', 'r2']].values

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

def batch_predict(β0, β1, β2, λ11, λ12, θ1, λ21, λ22, θ2, features, datetimeindex, n, return_numpy=False):
    # create sliding window of size n for r1 and r2 with shape (len(df) - n + 1, n)
    r1_sliding_window = torch.tensor(np.lib.stride_tricks.sliding_window_view(features[:, :, 0], n, axis=-1), dtype=torch.float64, requires_grad=False)
    r2_sliding_window = torch.tensor(np.lib.stride_tricks.sliding_window_view(features[:, :, 1], n, axis=-1), dtype=torch.float64, requires_grad=False)

    # calculate time difference between prediction datetime and each row in the sliding window with shape (len(df) - n + 1, n)
    dt = ((datetimeindex[1:] - datetimeindex[:-1]).days / 365).values
    dt_sliding_window = np.lib.stride_tricks.sliding_window_view(dt, n-1)
    dt_sliding_window = np.flip(dt_sliding_window, axis=1)
    dt_sliding_window = np.cumsum(dt_sliding_window, axis=1)
    dt_sliding_window = np.flip(dt_sliding_window, axis=1)
    dt_sliding_window = np.concatenate((dt_sliding_window, np.zeros((dt_sliding_window.shape[0], 1))), axis=1)
    t = torch.tensor(dt_sliding_window, dtype=torch.float64, requires_grad=False).unsqueeze(0)

    preds = β0 + β1 * R1(λ11, λ12, θ1, r1_sliding_window, t) + β2 * R2(λ21, λ22, θ2, r2_sliding_window, t)

    return preds.numpy() if return_numpy else preds

def residual(x, df, n, return_y_hat=False):
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
    if return_y_hat:
        return y - y_hat.numpy(), y_hat.numpy()
    else:
        return y - y_hat.numpy()

def torch_predict(params, x):
    β0, β1, β2, λ11, λ12, θ1, λ21, λ22, θ2 = params
    r1_sliding_window = x[:, 0, :]
    r2_sliding_window = x[:, 1, :]
    t = x[:, 2, :]
    return β0 + β1 * R1(λ11, λ12, θ1, r1_sliding_window, t) + β2 * R2(λ21, λ22, θ2, r2_sliding_window, t)

def evaluate(params, data, window):
    '''
    Evaluate the PDV model calculated using the sliding window of size window
    Prints the mean, min, max, mean absolute error, mean squared error and R^2 of the residuals
    params:
    params: list
        list of parameters of the PDV model
    data: pandas.DataFrame
        dataframe containing the data with columns 'r1', 'r2', 'vix' where 'r1' is the simple return, 'r2' is the squared return and 'vix' is the VIX level to be predicted
    window: int
        sliding window size to calculate R1 and R2
    '''
    residuals = residual(params, data, window)
    total_sum_of_squares = np.sum((data.iloc[window-1:]['vix'] - np.mean(data.iloc[window-1:]['vix'])) ** 2)
    print('Mean: {:.4f}, Min: {:.4f}, Max: {:.4f}, MAE: {:.4f}, MSE: {:.4f}, R^2: {:.4f}'.format(
        np.mean(residuals), np.min(residuals), np.max(residuals), np.mean(np.abs(residuals)), np.mean(residuals**2), (1-np.sum(residuals**2)/total_sum_of_squares)))

def plot(params, data, window, valid_start_date=None, test_start_date=None):
    '''
    Plot the VIX level and the predicted VIX level using the PDV model calculated using the sliding window of size window
    params:
    params: list
        list of parameters of the PDV model
    data: pandas.DataFrame
        dataframe containing the data with columns 'r1', 'r2', 'vix' where 'r1' is the simple return, 'r2' is the squared return and 'vix' is the VIX level to be predicted
    window: int
        sliding window size to calculate R1 and R2
    valid_start_date: datetime.date
        start date of the validation period
    test_start_date: datetime.date
        start date of the test period
    '''
    preds = predict(*params, data, window)
    df_preds = data[window-1:].copy()
    df_preds['preds'] = preds
    df_preds[['vix', 'preds']].plot(figsize=(15, 10))
    if valid_start_date is not None:
        plt.vlines(valid_start_date, plt.gca().get_ylim()[0], plt.gca().get_ylim()[-1], linestyles='dashed', colors='orange')
    if test_start_date is not None:
        plt.vlines(test_start_date, plt.gca().get_ylim()[0], plt.gca().get_ylim()[-1], linestyles='dashed', colors='red')

class PDV2Exp():

    def __init__(self, params):
        if isinstance(params, list):
            if len(params) == 9:
                self.β0, self.β1, self.β2, self.λ11, self.λ12, self.θ1, self.λ21, self.λ22, self.θ2 = params
                self.params = [self.β0, self.β1, self.β2, self.λ11, self.λ12, self.θ1, self.λ21, self.λ22, self.θ2]
            else:
                raise ValueError('params must have length 9 in the order: β0, β1, β2, λ11, λ12, θ1, λ21, λ22, θ2')
        else:
            raise TypeError('params must be a list in the order: β0, β1, β2, λ11, λ12, θ1, λ21, λ22, θ2')

    def predict(self, df, n):
        '''
        See predict function above
        '''
        return predict(self.β0, self.β1, self.β2, self.λ11, self.λ12, self.θ1, self.λ21, self.λ22, self.θ2, df, n)

    def residual(self, df, n):
        '''
        See residual function above
        '''
        return residual(self.params, df, n)

    def optimise(self, df, n, lower_bound=None, upper_bound=None, return_result=False):
        '''
        Optimise the parameters of the PDV model using least squares
        params:
        df: pandas.DataFrame
            dataframe containing the data with columns 'r1', 'r2', 'vix' where 'r1' is the simple return, 'r2' is the squared return and 'vix' is the VIX level to be predicted
        n: int
            sliding window size to calculate R1 and R2
        lower_bound: list
            list of lower bounds of the parameters of the PDV model
        upper_bound: list
            list of upper bounds of the parameters of the PDV model
        '''
        if lower_bound is None:
            lower_bound = [-np.inf] * 9
        if upper_bound is None:
            upper_bound = [np.inf] * 9

        res = least_squares(residual, self.params, args=(df, n), bounds=(lower_bound, upper_bound), verbose=2, ftol=1e-6)
        self.params = res.x

        if return_result:
            return res

    def evaluate(self, df, n):
        '''
        See evaluate function above
        '''
        evaluate(self.params, df, n)

    def plot(self, df, n, valid_start_date=None, test_start_date=None):
        '''
        See plot function above
        '''
        plot(self.params, df, n, valid_start_date, test_start_date)

    def batch_predict(self, features, datetimeindex, n, return_numpy=False):
        return batch_predict(*self.params, features, datetimeindex, n, return_numpy)
