import os
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, dirichlet
from scipy.special import logsumexp
from hmmlearn import hmm

def logdotexp(A, B):
    max_A = np.max(A)
    max_B = np.max(B)
    C = np.dot(np.exp(A - max_A), np.exp(B - max_B))
    np.log(C, out=C)
    C += max_A + max_B
    return C

def cal_logprob_obs_state(obs, μ, σ, dt, ε=1e-6, floor=1e-200):
    means = (μ[np.newaxis,:] - 0.5 * σ[np.newaxis,:]**2) * dt[:,np.newaxis]
    stds = σ[np.newaxis,:] * np.sqrt(dt[:,np.newaxis])
    # log_probs = norm.logpdf(obs[:,np.newaxis], means, stds) # (n_obs, 1), (n_states, 1), (n_states, 1) -> (n_obs, n_states)
    probs = np.maximum(norm.cdf(obs[:,np.newaxis] + ε, means, stds) - norm.cdf(obs[:,np.newaxis] - ε, means, stds), floor)
    log_probs = np.log(probs)
    return log_probs

def from_np_array(array_string):
    array_string = array_string.replace('[  ', '[')
    array_string = array_string.replace('[ ', '[')
    array_string = ','.join(array_string.split())
    try:
        return np.array(ast.literal_eval(array_string))
    except ValueError:
        raise ValueError(f'Unable to convert {array_string} to numpy array')

def plot_regimes(preds, date_index):
    '''
    Function to plot the regimes on the underlying path
    '''

    # get regime change indices
    change_idx = list([0]) + list(np.where(preds[1:] != preds[:-1])[0] + 1)
    # create color scheme as list of len equal to number of regimes
    colors = ['red', 'green', 'yellow', 'orange', 'cyan', 'pink', 'purple']
    # fill between each regime based on color scheme
    for i in range(len(change_idx)-1):
        plt.axvspan(date_index[change_idx[i]], date_index[change_idx[i+1]], alpha=0.2, color=colors[preds[change_idx[i]]])
    plt.axvspan(date_index[change_idx[-1]], date_index[-1], alpha=0.2, color=colors[preds[change_idx[-1]]])

class IrregularPeriodsHMM():
    '''
    Class to define a HMM for the S&P500 index
    Assumes S&P500 follows a regime switching geometric Brownian motion
    Each regime has a different drift and volatility
    '''
    def __init__(self, n_states, μ=None, σ=None, π_alpha=None, A_alpha=None, verbose=True):
        self.n_states = n_states
        self.μ = np.random.normal(0.1, 0.1, size=n_states) if μ is None else np.array(μ, dtype=np.float64)
        self.σ = np.random.uniform(0.2, 0.4, size=n_states) if σ is None else np.array(σ, dtype=np.float64)
        self.A_alpha = np.ones((n_states, n_states)) if A_alpha is None else np.array(A_alpha, dtype=np.float64)
        self.A = np.zeros((n_states, n_states))
        for i in range(n_states):
            self.A[i,:] = np.log(dirichlet.rvs(alpha=self.A_alpha[i,:], size=1))
        self.π_alpha = np.ones(n_states) if π_alpha is None else np.array(π_alpha, dtype=np.float64)
        self.π = np.log(dirichlet.rvs(self.π_alpha, size=1))
        self.verbose = verbose

    def forward(self):
        '''
        Function to calculate the forward probabilities
        '''
        T = len(self.obs)
        K = self.n_states
        alpha = np.empty((T,K))
        logprob_obs_t_hist = np.empty((T))

        alpha[0] = self.logprob_obs_state[0] + self.π
        alpha[0] -= logsumexp(alpha[0]) # scaling term
        for t in range(1, T):
            alpha[t] = logdotexp(alpha[t-1], self.A) + self.logprob_obs_state[t]
            logprob_obs_t_hist[t] = logsumexp(alpha[t]) # scaling term
            alpha[t] -= logprob_obs_t_hist[t]

        return alpha, logprob_obs_t_hist

    def backward(self, logprob_obs_t_hist):
        '''
        Function to calculate the backward probabilities
        '''
        T = len(self.obs)
        K = self.n_states
        beta = np.zeros((T,K))
        for t in range(T-2, -1, -1):
            beta[t] = logdotexp(self.A, (self.logprob_obs_state[t+1] + beta[t+1]))
            beta[t] -= logprob_obs_t_hist[t+1]
        return beta

    def check_convergence(self, log_likelihood, tol):
        '''
        Function to check for convergence
        '''
        if self.iteration > 0:
            if np.isnan(log_likelihood):
                print('Log likelihood is NaN')
                return True
            if log_likelihood < self.log_likelihood:
                print(f'Log likelihood decreased from {self.log_likelihood} to {log_likelihood}')
                return True
            if np.abs((log_likelihood - self.log_likelihood) / self.log_likelihood) < tol:
                self.log_likelihood = log_likelihood
                print(f'Iteration {self.iteration + 1}: log likelihood = {log_likelihood}')
                print('Convergence attained')
                return True
        self.log_likelihood = log_likelihood
        if self.verbose:
            print(f'Iteration {self.iteration + 1}: log likelihood = {log_likelihood}')
        self.iteration += 1
        return False

    def EM_optimise(self, df, tol=1e-6):
        '''
        Function to optimise the parameters of the HMM using the EM algorithm
        '''
        self.df = df
        df['r1'] = np.log(df['Close']).diff()
        self.dt = ((df.index[1:] - df.index[:-1]).days / 365).values
        df.dropna(inplace=True)
        self.obs = df['r1'].values

        T = len(self.obs)
        K = self.n_states

        self.iteration = 0 # iteration counter for EM algorithm
        while True:
            # E STEP

            # Calculate log P(obs|state) for each state
            self.logprob_obs_state = cal_logprob_obs_state(self.obs, self.μ, self.σ, self.dt)

            # calculate the forward and backward probabilities
            alpha, logprob_obs_t_hist = self.forward()
            beta = self.backward(logprob_obs_t_hist)

            # calculate the posterior P(state|obs)
            self.logprob_state_obs = alpha + beta
            self.q_prob = np.exp(self.logprob_state_obs) # (T, K)
            assert np.allclose(np.sum(self.q_prob, axis=-1), 1)

            # calculate the transitional posterior P(state(t-1), state(t)|obs)
            self.logprob_state_trans = np.zeros((T-1,K,K))
            for t in range(T-1):
                self.logprob_state_trans[t,:,:] = np.tile(alpha[t,:], (K,1)).T + self.A + np.tile(self.logprob_obs_state[t+1,:], (K,1)) + np.tile(beta[t+1,:], (K,1))
                self.logprob_state_trans[t,:,:] -= logprob_obs_t_hist[t+1][...,np.newaxis,np.newaxis] # normalization
            assert np.allclose(np.sum(np.exp(self.logprob_state_trans), axis=(1,2)), 1)

            # CHECK FOR CONVERGENCE
            if self.check_convergence(np.sum(logprob_obs_t_hist), tol):
                break

            # M STEP
            # update pi parameters using MAP
            self.π_alpha += self.q_prob[0]

            # update pi log prob using MAP with the parameters
            self.π = (self.π_alpha - 1) / (np.sum(self.π_alpha) - K)
            self.π = np.log(self.π)

            # update A parameters using MAP
            self.A_alpha += np.sum(np.exp(self.logprob_state_trans), axis=0)

            # update A log prob using MAP with the parameters
            self.A = (self.A_alpha - 1) / (np.sum(self.A_alpha, axis=-1) - K)[...,np.newaxis] # to align the division
            self.A = np.log(self.A)

            # update μ and σ using MLE
            drift = (self.q_prob * self.obs[:,np.newaxis] / self.dt[:,np.newaxis]).sum(axis=0) / self.q_prob.sum(axis=0)
            self.σ = np.sqrt((self.q_prob * (self.obs[:,np.newaxis] - drift[np.newaxis,:] * self.dt[:,np.newaxis])**2 / self.dt[:,np.newaxis]).sum(axis=0) / self.q_prob.sum(axis=0))
            self.μ = drift + 0.5 * self.σ**2

    def predict(self, df):
        self.df = df
        df['r1'] = np.log(df['Close']).diff()
        self.dt = ((df.index[1:] - df.index[:-1]).days / 365).values
        df.dropna(inplace=True)
        self.obs = df['r1'].values

        self.logprob_obs_state = cal_logprob_obs_state(self.obs, self.μ, self.σ, self.dt)

        # calculate the forward and backward probabilities
        alpha, logprob_obs_t_hist = self.forward()
        beta = self.backward(logprob_obs_t_hist)

        # calculate the posterior P(state|obs)
        self.logprob_state_obs = alpha + beta
        self.q_prob = np.exp(self.logprob_state_obs) # (T, K)

    def print_params(self, precision=3):
        '''
        Function to print the parameters of the HMM
        '''
        print(f'μ: {np.array2string(self.μ, precision=precision)}')
        print(f'σ: {np.array2string(self.σ, precision=precision)}')
        print(f'π: {np.array2string(np.exp(self.π), precision=precision)}')
        print(f'A:\n{np.array2string(np.exp(self.A), precision=precision)}')

    def save_params_to_csv(self, path, model_name):
        '''
        Function to save the parameters of the HMM to a csv file
        '''
        df = pd.DataFrame({'n_regimes': self.n_states, 'model': model_name, 'log_likelihood': self.log_likelihood,
                           'μ': [self.μ], 'σ': [self.σ],
                           'π': [np.exp(self.π)], 'A': [np.exp(self.A)],
                           'π_alpha': [self.π_alpha], 'A_alpha': [self.A_alpha]})
        df.to_csv(path, mode='a', header=(not os.path.exists('params.csv')))

    def load_params_from_csv(self, path, model_name):
        '''
        Function to load the parameters of the HMM from a csv file
        '''
        df = pd.read_csv(path, index_col=0, converters={'μ': from_np_array, 'σ': from_np_array,
                                                        'π': from_np_array, 'A': from_np_array,
                                                        'π_alpha': from_np_array, 'A_alpha': from_np_array})
        df = df[(df['n_regimes'] == self.n_states) & (df['model'] == model_name)]
        if len(df) == 1:
            self.μ = df['μ'].values[0]
            self.σ = df['σ'].values[0]
            self.π = np.log(df['π'].values[0])
            self.A = np.log(df['A'].values[0])
            try:
                self.π_alpha = df['π_alpha'].values[0]
                self.A_alpha = df['A_alpha'].values[0]
            except:
                self.π_alpha = np.exp(self.π) * 1e6
                self.A_alpha = np.exp(self.A) * 1e6
            self.log_likelihood = df['log_likelihood'].values[0]
            print(f'Loaded parameters from {path}')
        elif len(df) == 0:
            raise ValueError(f'No parameters found for n_regimes={self.n_states} and model={model_name}')
        else:
            raise ValueError(f'Multiple parameters found for n_regimes={self.n_states} and model={model_name}')

    def load_df(self, df):
        self.df = df

    def plot_regimes_on_index(self):
        '''
        Function to plot the regimes on the underlying path
        '''
        plt.figure(figsize=(16,9))

        try:
            self.df['Close'].plot()
        except AttributeError:
            print('Perform EM_optimise() on data or load_df() first')

        preds = self.q_prob.argmax(axis=1)
        plot_regimes(preds, self.df.index)
        plt.legend()
        plt.show()

    def simulate(self, return_df=False):
        '''
        Function to simulate the underlying path
        '''
        try:
            dt = ((self.df.index[1:] - self.df.index[:-1]).days / 365).values
        except AttributeError:
            print('Perform EM_optimise() on data or load_df() first')

        μ = self.μ
        σ = self.σ
        π = np.exp(self.π)
        A = np.exp(self.A)

        # simulate geometric brownian motion
        state = np.random.choice(self.n_states, p=π)
        states = [state]
        px = [self.df.loc[self.df.index[0], 'Close']]
        for i in range(len(dt)):
            px.append(px[-1] * np.exp((μ[state] - σ[state]**2/2) * dt[i] + σ[state] * np.sqrt(dt[i]) * np.random.randn()))
            state = np.random.choice(self.n_states, p=A[state])
            states.append(state)
        px = np.array(px)
        plt.figure(figsize=(16,9))
        plt.plot(self.df.index, px)
        plot_regimes(np.array(states), self.df.index)
        plt.show()
        sim_df = pd.DataFrame({'Close': px}, index=self.df.index)
        if return_df:
            return sim_df

class RegularPeriodsHMM():

    def __init__(self,
                 n_states,
                 n_models,
                 covariance_type='full',
                 min_covar=0.001,
                 means_prior=0.,
                 covars_prior=0.01,
                 startprob_prior=1.0,
                 transmat_prior=1.0,
                 tol=1e-6,
                 n_iter=100,
                 verbose=True):
        self.n_states = n_states
        self.n_models = n_models
        self.verbose = verbose
        self.models = []
        for _ in range(n_models):
            model = hmm.GaussianHMM(n_components=n_states,
                                    covariance_type=covariance_type,
                                    min_covar=min_covar,
                                    means_prior=means_prior,
                                    covars_prior=covars_prior,
                                    startprob_prior=startprob_prior,
                                    transmat_prior=transmat_prior,
                                    n_iter=n_iter,
                                    tol=tol)
            # model.init_params = ''
            self.models.append(model)

    def fit(self, data, col:list=['Close']):

        # get log returns from time series data which can have multiple columns
        x = (np.log(data[col] / data[col].shift(1)).dropna()).values

        # fit each model and choose the best model
        high_score = -np.inf
        for i in range(self.n_models):
            self.models[i].fit(x)
            preds = self.models[i].predict(x)
            print(f'Model score: {self.models[i].score(x):0.4f}')
            model_score = self.models[i].score(x)
            if model_score > high_score:
                self.best_model = i
                high_score = model_score
                best_preds = preds
        print(f'Best model: {self.best_model}')

        # print number of observations in each regime
        for i in range(self.n_states):
            print(f'Regime {i} count: {(best_preds == i).sum()}')

        # plot the underlying path and regimes
        data[col].plot(figsize=(16, 9))
        plot_regimes(best_preds, data.index)

        # get parameters of best model
        self.drift = self.models[self.best_model].means_.squeeze()
        self.covar = (self.models[self.best_model].covars_.squeeze())
        self.transition = self.models[self.best_model].transmat_

        if x.shape[-1] == 1:
            self.sigma = np.sqrt(self.covar*252)
        else:
            self.sigma = []
            for i in range(self.n_states):
                self.sigma.append(np.sqrt(self.covar[i].diagonal()*252))
            self.sigma = np.array(self.sigma)
        self.mu = (self.drift*252) + self.sigma**2/2
        self.pi = self.models[self.best_model].startprob_
        print(f'μ: {self.mu}')
        print(f'σ: {self.sigma}')
        print(f'Drift: {self.mu-self.sigma**2/2}')
        print('Transition:')
        print(self.transition)
        print(f'Initial state distribution: {self.pi}')

    def simulate(self, start_value, indices, steps_per_unit_time=252, seed=None):

        # training could have been done on multiple time series but simulation is only done on first time series
        if self.drift.ndim == 2:
            drift = self.drift[:,0]
            sigma = self.sigma[:,0]
        else:
            drift = self.drift
            sigma = self.sigma

        # simulate geometric brownian motion
        rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
        px = [start_value]
        state = rng.choice(np.arange(self.n_states), p=self.pi)
        states = [state]
        for _ in range(len(indices)-1):
            px.append(px[-1] * np.exp((drift[state])/steps_per_unit_time + sigma[state]*rng.normal()/np.sqrt(steps_per_unit_time)))
            state = rng.choice(np.arange(self.n_states), p=self.transition[state])
            states.append(state)

        # create dataframe of simulated path
        sim_df = pd.DataFrame({'Close': px}, index=indices)

        # plot simulated path
        states = np.array(states)
        sim_df.plot(figsize=(16, 9))
        plot_regimes(states, indices)

        return sim_df

    def batch_simulate(self, n_samples, start_value, indices, steps_per_unit_time=252, seed=None):

        # training could have been done on multiple time series but simulation is only done on first time series
        if self.drift.ndim == 2:
            drift = self.drift[:,0]
            sigma = self.sigma[:,0]
        else:
            drift = self.drift
            sigma = self.sigma

        # simulate geometric brownian motion
        rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
        px = np.ones((n_samples, len(indices))) * start_value
        state = rng.multinomial(n=1, pvals=self.pi, size=n_samples).argmax(axis=1)
        # states = [state]
        for i in range(len(indices)-1):
            px[:,i+1] = px[:,-1] * np.exp((drift[state])/steps_per_unit_time + sigma[state]*rng.normal(n_samples)/np.sqrt(steps_per_unit_time))
            state = rng.multinomial(n=1, pvals=self.transition[state], size=n_samples).argmax(axis=1)
            # state.append(state)

        return px
