import matplotlib.pyplot as plt
import utils as NMT
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad, cumulative_trapezoid

class GaussianBinaryTest:
    
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    
    def beta0(self, tau):
        return tau
    
    def beta1(self, tau):
        return NMT.tau_to_beta(tau, self.mu, self.sigma)
    
    def beta1_prime(self, tau):
        quantile = norm.ppf(1-tau)
        numerator = norm(0,1).pdf(quantile - self.mu)
        denominator = norm(0,1).pdf(quantile)
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    
    def beta1_prime_threshold(self):
        small_tau = np.linspace(0,0.001,100)[1:-1]
        big_tau = np.linspace(0,1,1000)[1:-1]
        tau_range = np.concatenate((small_tau, big_tau))
        beta1_prime_value = self.beta1_prime(tau_range)
        return tau_range[beta1_prime_value > 1]
        
    def utility(self, pi, tau, reward, cost):
        beta1 = self.beta1(tau)
        beta0 = self.beta0(tau)
        approval_prob = pi * beta0 + (1 - pi) * beta1
        return reward * approval_prob - cost
    
    def compute_fdr(self, pi, tau):
        beta1 = self.beta1(tau)
        beta0 = self.beta0(tau)
        numerator = beta0 * pi
        denominator = beta0 * pi + beta1 * (1 - pi)
        return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)
    
    def optimal_threshold(self, pi, fdr_level, num_tau=1000):
        tau_range = np.linspace(0, 1, num_tau)
        fdr_range = self.compute_fdr(pi, tau_range)
        idx = np.where(fdr_range <= fdr_level)[0][-1]
        return tau_range[idx]
    
    def optimal_threshold_to_type(self, tau, fdr_level, num_type=1500):
        type_range = np.linspace(0, 1, num_type)
        fdr_range = self.compute_fdr(type_range, tau)
        idx = np.where(fdr_range <= fdr_level)[0][-1]
        return type_range[idx]
    

    def build_two_contract(self, pi_1, pi_2, tau1, tau2, reward2, cost2, epsilon):
        beta0_tau1 = self.beta0(tau1)
        beta1_tau1 = self.beta1(tau1)
        beta0_tau2 = self.beta0(tau2)
        beta1_tau2 = self.beta1(tau2)

        Delta_1 = beta1_tau1 - beta0_tau1
        Delta_2 = beta1_tau2 - beta0_tau2

        reward1 = reward2 * (Delta_2 / Delta_1) + epsilon

        cost_const = reward1 * beta1_tau1 - reward2 * beta1_tau2 + cost2
        cost1_lb = pi_2 * (reward2 * Delta_2 - reward1 * Delta_1) + cost_const
        cost1_ub = pi_1 * (reward2 * Delta_2 - reward1 * Delta_1) + cost_const

        return reward1, cost1_lb, cost1_ub
    
    
    def build_sep_menu(self, pi_list, tau_list, base_reward, base_cost, epsilon, cost_position):
        num_type = pi_list.size
        reward_list = np.zeros(num_type)
        cost_list = np.zeros(num_type)

        reward_list[-1] = base_reward
        cost_list[-1] = base_cost

        for t in range(num_type-1, 0, -1):
            pi_1 = pi_list[t-1]
            pi_2 = pi_list[t]
            tau1 = tau_list[t-1]
            tau2 = tau_list[t]
            reward2 = reward_list[t]
            cost2 = cost_list[t]

            reward1, cost1_lb, cost1_ub = self.build_two_contract(
                pi_1, pi_2, tau1, tau2, reward2, cost2, epsilon
            )
            reward_list[t-1] = reward1

            ## different cost_choice

            if cost_position == "Lb":
                cost_list[t-1] = cost1_lb + 0.01
            elif cost_position == "Ub":
                cost_list[t-1] = cost1_ub - 0.01
            else:
                cost_list[t-1] = (cost1_lb + cost1_ub)/2

        return tau_list, reward_list, cost_list