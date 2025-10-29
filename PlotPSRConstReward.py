
import matplotlib.pyplot as plt
import utils as NMT
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad, cumulative_trapezoid
from GaussianBinaryTest import *

fig_directory = "./Figure/"


def compute_G_constR(test, reward, fdr_level):
    valid_tau = test.beta1_prime_threshold()
    valid_type = np.array([test.optimal_threshold_to_type(valid_tau[i], fdr_level = fdr_level) for i in range(len(valid_tau))])
    integrand_value = test.beta1(valid_tau) - test.beta0(valid_tau)
    cum_integral = cumulative_trapezoid(integrand_value, valid_type, initial=0)
    total_intergal = cum_integral[0]
    G_value = (total_intergal - cum_integral) * reward
    return valid_tau, valid_type, G_value


# Setup your tests and parameters
mu_values = [0.5, 1, 2]  # extend this list as needed
const_reward = 100
fdrlevel = 0.25
colors = plt.cm.tab10.colors

# Create tests
tests = [GaussianBinaryTest(mu=mu, sigma=1.0) for mu in mu_values]

######## Plot Configuration ########

fig_height = 5
fig_width = 6

axislabel_fontsize = 15
axistick_fontsize = 15
legend_fontsize = 13
fontmed = 16
fontlarge = 20

####### Start Plot ############


for i, (test, mu) in enumerate(zip(tests, mu_values)):
    valid_tau, valid_type, G_value = compute_G_constR(test, const_reward, fdrlevel)
    
    ### The current plots have an artificial upper bound due to finite grid of threshold values we use
    ### Manually add threshold 0 for type 1 with G(1) = 0 to remove this upper bound

    valid_tau = np.concatenate(([0], valid_tau))
    valid_type = np.concatenate(([1], valid_type))
    G_value = np.concatenate(([0], G_value))
    
    
    color = colors[i % len(colors)]  # cycle colors if more than 10 tests
    

    # Prepare the plot
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.plot(valid_type, G_value, color=color, label=fr"$\theta_1 = {mu}$")
    plt.fill_between(valid_type, G_value, alpha=0.3, color=color)

    # Configure ticks for the x-axis based on first test (or adjust if needed)
    ticks = np.linspace(valid_type.min(), valid_type.max(), num=5)
    plt.xticks(ticks=ticks, labels=[f"{x:.2f}" for x in ticks])

    # Add the static text box (optional, you might want to move or customize this)
    plt.text(
        0.06, 0.05, 'Information Rent',
        transform=plt.gca().transAxes,
        fontsize=fontlarge, color='blue', weight='bold',
        ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='none')
    )

    # Add a legend to identify the curves instead of repeated text boxes
    plt.legend(fontsize=legend_fontsize, loc='upper right')

    plt.xlabel('Agent type', fontsize=fontmed)
    plt.ylabel('G value', fontsize=fontmed)
    plt.title('Optimal G with fixed rewards', fontsize=fontlarge)
    plt.grid(linestyle='--', alpha=0.5)
    plt.tick_params(axis='both', labelsize=axistick_fontsize)
    plt.tight_layout()

    figname = f"PSRMenu_constR_mean_{mu}.pdf"
    plt.savefig(fig_directory + figname, format="pdf", bbox_inches="tight")
    
    plt.show()
