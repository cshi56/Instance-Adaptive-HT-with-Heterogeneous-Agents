import matplotlib.pyplot as plt
import utils as NMT
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad, cumulative_trapezoid
from GaussianBinaryTest import *

fig_directory = "./Figure/"


# Define test configurations
mu_values = [0.5, 1, 2]
labels = [r"$\theta_1 = 0.5$", r"$\theta_1 = 1$", r"$\theta_1 = 2$"]
colors = plt.cm.tab10.colors

# Initialize tests
tests = [GaussianBinaryTest(mu=mu, sigma=1.0) for mu in mu_values]

# Precompute thresholds and power values at those thresholds
thresholds = [test.beta1_prime_threshold()[-1] for test in tests]
powers = [test.beta1(tau) for test, tau in zip(tests, thresholds)]


######## Plot Configuration ########

fig_height = 5
fig_width = 6

axislabel_fontsize = 15
axistick_fontsize = 15
legend_fontsize = 13
fontmed = 16
fontlarge = 20

####### Start Plot ############


# Set up plot
fig = plt.figure(figsize=(fig_width, fig_height))
tau_range = np.linspace(0, 1, 1000)[:-1]

# Plot curves and mark threshold points
for i, (test, label, color, tau, power) in enumerate(zip(tests, labels, colors, thresholds, powers)):
    plt.plot(tau_range, test.beta1(tau_range), color=color, label=label)
    plt.vlines(tau, 0, power, color=color, linestyle='--', linewidth=1)
    plt.scatter([tau], [power], color=color, zorder=5)
    plt.scatter([tau], [0], color=color, zorder=3)
    plt.annotate(f"({tau:.2f}, {power:.2f})", (tau, power),
                 textcoords="offset points", xytext=(5, 5),
                 color=color, fontsize=fontmed)

# Final plot decorations
plt.xlabel(r"$p$-value threshold $\tau$", fontsize=fontmed)
plt.ylabel(r"$\beta_1(\tau)$", fontsize=fontmed)
plt.title(r"Power function for various $\theta_1$", fontsize=fontlarge)
plt.legend(loc="lower right", fontsize=legend_fontsize)
plt.grid(linestyle='--', alpha=0.5)
plt.tick_params(axis='both', labelsize=axistick_fontsize)
plt.tight_layout()

# Save and display
figname = "PowerCurve_GMT.pdf"
plt.savefig(fig_directory + figname, format="pdf", bbox_inches="tight")

plt.show()