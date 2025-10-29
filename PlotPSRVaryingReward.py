import matplotlib.pyplot as plt
import utils as NMT
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad, cumulative_trapezoid
from GaussianBinaryTest import *

fig_directory = "./Figure/"


Test = GaussianBinaryTest(mu=1, sigma=1.0)
worst_type = 0.8
worst_reward = 100
fdr_level = 0.25

worst_threshold = Test.optimal_threshold(worst_type, fdr_level, num_tau=1000)
worst_cost = worst_reward * (worst_type * worst_threshold + (1-worst_type) * Test.beta1(worst_threshold))
#print(worst_cost)


def epsilon(z, delta):
    return delta * (1 - z) ** 2

def integrand(z, delta):
    return 1 + epsilon(z, delta)


def compute_G_varyingR(test, worst_type, worst_reward, integrand_func, delta, fdr_level):
    worst_threshold = test.optimal_threshold(worst_type, fdr_level, num_tau=1000)
    #worst_cost = worst_reward * (worst_type * worst_threshold + (1-worst_type) * test.beta1(worst_threshold))
    
    type_range = np.linspace(0, worst_type, 1000)
    z_range = np.linspace(0, worst_type, 1000)
    integrand_value = integrand_func(z_range, delta)
    cum_integrals = cumulative_trapezoid(integrand_value, type_range, initial=0)
    total_integral = cum_integrals[-1]
    G_value = (total_integral - cum_integrals) * worst_reward * (test.beta1(worst_threshold) - test.beta0(worst_threshold))
    
    return worst_threshold, type_range, G_value


G_value_list = []
delta_list = [0.01, 0.1, 0.5, 1]
for j in delta_list:
    worst_threshold, type_range, G_val = compute_G_varyingR(test=Test, worst_type=worst_type, worst_reward=worst_reward, integrand_func = integrand, delta = j, fdr_level=fdr_level)
    G_value_list.append(G_val)

base_contract_utility = Test.utility(type_range, worst_threshold, worst_reward, worst_cost)


######## Plot Configuration #########

fig_height = 5
fig_width = 8

axislabel_fontsize = 15
axistick_fontsize = 15
legend_fontsize = 13
fontmed = 16
fontlarge = 20


######### Start Plot #############

fig = plt.figure(figsize=(fig_width, fig_height))

colors = plt.cm.tab10.colors

#plt.plot(type_range, base_contract_utility, label = "base contract", color = colors[0])
#plt.fill_between(type_range, base_contract_utility, alpha=0.3, color = colors[0])
for j in range(len(delta_list)):
    plt.plot(type_range, base_contract_utility-G_value_list[j], label = fr"$\eta$ = {delta_list[j]}", color = colors[j])

plt.fill_between(type_range, base_contract_utility - G_value_list[0], alpha=0.3, color = colors[0])
    
    
#plt.text(
#    0.06, 0.05, 'Information Rent',
#    transform=plt.gca().transAxes,
#    fontsize=fontlarge, color='blue', weight='bold',
#    ha='left', va='bottom',
#    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='none')
#)

plt.xlabel('Agent type', fontsize = fontmed)
plt.ylabel('Expected return', fontsize = fontmed)
plt.title("Optimal G for principal's expected return", fontsize = fontlarge)
plt.legend(loc = "lower right", fontsize = legend_fontsize)
plt.grid(linestyle = '--', alpha=0.5)
plt.tick_params(axis='both', labelsize = axistick_fontsize)
plt.tight_layout()

figname = "PSRMenu_varyingR.pdf"
plt.savefig(fig_directory + figname, format="pdf", bbox_inches="tight")

plt.show()