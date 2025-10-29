
import matplotlib.pyplot as plt
import utils as NMT
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad, cumulative_trapezoid
from GaussianBinaryTest import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import MaxNLocator


fig_directory = "./Figure/"


def compute_G_constR(test, reward, fdr_level):
    valid_tau = test.beta1_prime_threshold()
    valid_type = np.array([test.optimal_threshold_to_type(valid_tau[i], fdr_level = fdr_level) for i in range(len(valid_tau))])
    integrand_value = test.beta1(valid_tau) - test.beta0(valid_tau)
    cum_integral = cumulative_trapezoid(integrand_value, valid_type, initial=0)
    total_intergal = cum_integral[0]
    G_value = (total_intergal - cum_integral) * reward
    return valid_tau, valid_type, G_value



def compute_gap(true_test, mis_test, valid_tau, valid_type):
    
    num = valid_type + (1-valid_type) * true_test.beta1_prime(valid_tau) - mis_test.beta1_prime(valid_tau)
    denom = 1 - true_test.beta1_prime(valid_tau)
    map_type = num/denom
    
    idx = np.where((map_type > 0) & (map_type < 1))[0]
    final_type = valid_type[idx]
    final_tau = valid_tau[idx]
    
    termA = ((1 - final_type) / final_type) * true_test.beta1(final_tau)
    
    derivative_deviation = mis_test.beta1_prime(final_tau) - true_test.beta1_prime(final_tau)
    total_deviation = (derivative_deviation / (true_test.beta1_prime(final_tau) - 1))
    termB = (mis_test.beta1(final_tau) * (1 - final_type)) / (final_type + total_deviation)
    
    return final_type, termA, termB


true_test = GaussianBinaryTest(mu=1, sigma=1.0)
const_reward = 100
fdr_level = 0.25
valid_tau, valid_type, G_value = compute_G_constR(test = true_test, reward = const_reward, fdr_level = fdr_level)


###### Plot Configuration ######

fig_height = 5
fig_width = 8

axislabel_fontsize = 15
axistick_fontsize = 15
legend_fontsize = 13
fontmed = 16
fontlarge = 20

######## Start Plot for OverPower#########


fig, ax = plt.subplots(figsize=(fig_width, fig_height))
figname = "Misspecified_overpower.pdf"

colors = plt.cm.tab10.colors
mu_list = [1.1, 1.2, 1.3, 2.0]

### main plot ###

for i, mu in enumerate(mu_list):
    mis_test = GaussianBinaryTest(mu=mu, sigma=1.0)
    final_type, termA, termB = compute_gap(true_test, mis_test, valid_tau, valid_type)
    diff = termA - termB

    # baseline curve
    ax.plot(final_type, np.maximum(diff, -10), color=colors[i], alpha=0.5)

    # highlight violations
    idx = diff > 0
    ax.plot(final_type[idx], diff[idx], color=colors[i], lw=3,
            label=fr"$\theta_1={mu}$")

# legend only for the main axis

ax.legend(
    frameon=False,
    loc="lower right",   # or "lower center" if you want it away from curves
    ncol=2,             # <-- 2 columns
    columnspacing=1.0,  # tighten spacing between columns
    handletextpad=0.4,  # tighten spacing between line and label
    borderaxespad=0.8, # distance from the axes
    fontsize = legend_fontsize
)

ax.axhline(0, ls="--", lw=1, color="k", alpha=0.5)
ax.set_xlabel("Reported type",fontsize = fontmed)
ax.set_ylabel("FDR gap", fontsize = fontmed)
ax.set_title("Reported type where FDR Gap > 0", fontsize = fontlarge)

ax.tick_params(axis='both', labelsize = axistick_fontsize)

#### Zoom in plot ####

# inset: shifted a bit upward from bottom
axins = inset_axes(
    ax,
    width="45%", height="45%",
    loc="lower right",
    bbox_to_anchor=(-0.05, 0.3, 1, 1), 
    bbox_transform=ax.transAxes,
    borderpad=0.8,
)

for i, mu in enumerate(mu_list):
    mis_test = GaussianBinaryTest(mu=mu, sigma=1.0)
    final_type, termA, termB = compute_gap(true_test, mis_test, valid_tau, valid_type)
    diff = termA - termB
    idx = diff > 0
    axins.plot(final_type[idx], diff[idx], color=colors[i], lw=2)


axins.tick_params(axis="both", which="major", labelsize=int(0.9*axistick_fontsize))
axins.axhline(0, ls="--", lw=1, color="k", alpha=0.5)
axins.autoscale(enable=True, axis="both", tight=False)
axins.margins(x=0.05, y=0.1)
mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")

plt.savefig(fig_directory + figname, format="pdf", bbox_inches="tight")
plt.show()


############# Start Plot for underPower #########

figname = "Misspecified_underpower.pdf"
plt.figure(figsize=(fig_width, fig_height))

mu_list = [0.6, 0.7, 0.8, 0.9]

for i, mu in enumerate(mu_list):
    mis_test = GaussianBinaryTest(mu=mu, sigma=1.0)
    final_type, termA, termB = compute_gap(true_test, mis_test, valid_tau, valid_type)
    diff = termA - termB

    # baseline curve
    plt.plot(final_type, np.maximum(diff, -10), color=colors[i], alpha=0.5)

    # highlight violations
    idx = diff > 0
    plt.plot(final_type[idx], diff[idx], color=colors[i], lw=2,
             label=fr"$\theta_1={mu}$")

# legend
plt.legend(
    frameon=False,
    loc="upper right",
    ncol=2,
    columnspacing=1.0,
    handletextpad=0.4,
    borderaxespad=0.8,
    fontsize=legend_fontsize
)

plt.axhline(0, ls="--", lw=1, color="k", alpha=0.5)
plt.xlabel("Reported type", fontsize=fontmed)
plt.ylabel("FDR gap", fontsize=fontmed)
plt.title("Reported type where FDR Gap > 0", fontsize=fontlarge)
plt.tick_params(axis="both", labelsize=axistick_fontsize)
plt.grid(linestyle = '--', alpha=0.5)

ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # try 10 ticks

plt.savefig(fig_directory + figname, format="pdf", bbox_inches="tight")
plt.show()
