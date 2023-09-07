import numpy as np
import pandas as pd
from scipy.stats import norm, dirichlet
from scipy.special import logsumexp

def logdotexp(A, B):
    max_A = np.max(A)
    max_B = np.max(B)
    C = np.dot(np.exp(A - max_A), np.exp(B - max_B))
    np.log(C, out=C)
    C += max_A + max_B
    return C

def cal_logprob_obs_state(obs, μ, σ, dt, ε=1e-6, floor=1e-100):
    means = (μ[np.newaxis,:] - 0.5 * σ[np.newaxis,:]**2) * dt[:,np.newaxis]
    stds = σ[np.newaxis,:] * np.sqrt(dt[:,np.newaxis])
    # log_probs = norm.logpdf(obs[:,np.newaxis], means, stds) # (n_obs, 1), (n_states, 1), (n_states, 1) -> (n_obs, n_states)
    probs = np.maximum(norm.cdf(obs[:,np.newaxis] + ε, means, stds) - norm.cdf(obs[:,np.newaxis] - ε, means, stds), floor)
    log_probs = np.log(probs)
    return log_probs

def initial_logprob(π, logprob_initial_state_obs):
    return np.sum(np.exp(logprob_initial_state_obs) * π)

def transition_logprob(A, logprob_state_trans):
    return np.sum(np.exp(logprob_state_trans) * A[np.newaxis,...])

def emission_logprob(q_prob, logprob_obs_state):
    return np.sum(q_prob * logprob_obs_state)

class SPXHMM():
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
        T = len(self.obs)
        K = self.n_states
        beta = np.zeros((T,K))
        for t in range(T-2, -1, -1):
            beta[t] = logdotexp(self.A, (self.logprob_obs_state[t+1] + beta[t+1]))
            beta[t] -= logprob_obs_t_hist[t+1]
        return beta

    def check_convergence(self, log_likelihood, tol):
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

    def print_params(self, precision=3):
        print(f'μ:       {np.array2string(self.μ, precision=precision)}')
        print(f'σ:       {np.array2string(self.σ, precision=precision)}')
        # print(f'π_alpha: {np.array2string(self.π_alpha, precision=precision)}')
        print(f'π:       {np.array2string(np.exp(self.π), precision=precision)}')
        # print(f'A_alpha: \n{np.array2string(self.A_alpha, precision=precision)}')
        print(f'A:       \n{np.array2string(np.exp(self.A), precision=precision)}')
