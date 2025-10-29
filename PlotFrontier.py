import matplotlib.pyplot as plt
import numpy as np
from GaussianBinaryTest import GaussianBinaryTest


fig_directory = "./Figure/"

# Initialize test and parameters
Test = GaussianBinaryTest(mu=1, sigma=1.0)
type_g, type_b = 0.4, 0.7
prop_g, prop_b = 0.5, 0.5

# Tau grid
tau_grid = np.concatenate([np.linspace(0, 0.001, 200),
                           np.linspace(0.001, 0.1, 300),
                           np.linspace(0.1, 0.2, 200),
                           np.linspace(0.2, 1, 100)])

# Full 2D grid for FDR/TDR
tau_g_grid, tau_b_grid = np.meshgrid(tau_grid, tau_grid)
tdr = prop_g*(1-type_g)*Test.beta1(tau_g_grid) + prop_b*(1-type_b)*Test.beta1(tau_b_grid)
fdr = np.maximum(Test.compute_fdr(type_g, tau_g_grid), Test.compute_fdr(type_b, tau_b_grid))
fdr_flat, tdr_flat = fdr.ravel(), tdr.ravel()

# TDR/FDR for individual types
tdr_only_g = prop_g * (1-type_g) * Test.beta1(tau_grid)
tdr_only_b = prop_b * (1-type_b) * Test.beta1(tau_grid)
fdr_only_g = Test.compute_fdr(type_g, tau_grid)
fdr_only_b = Test.compute_fdr(type_b, tau_grid)

# Single-test scenario
tdr_single = tdr_only_g + tdr_only_b
fdr_single = np.maximum(fdr_only_g, fdr_only_b)

# Pareto frontier
tdr_list_menu, fdr_list_menu = [], []
alpha_grid = np.linspace(0, 1, 1000)
for alpha in alpha_grid:
    opt_tau_g = tau_grid[fdr_only_g <= alpha][-1]
    opt_tau_b = tau_grid[fdr_only_b <= alpha][-1]
    if opt_tau_b > 0:
        tdr_list_menu.append(prop_g*(1-type_g)*Test.beta1(opt_tau_g) + prop_b*(1-type_b)*Test.beta1(opt_tau_b))
        fdr_list_menu.append(max(Test.compute_fdr(type_g, opt_tau_g), Test.compute_fdr(type_b, opt_tau_b)))


######### Start Plot #############

######## Plot Configuration #########

fig_height = 5
fig_width = 8

axislabel_fontsize = 15
axistick_fontsize = 15
legend_fontsize = 13
fontmed = 16
fontlarge = 20


# -------------- A single plot -------------------

figname = "TwoAgentFrontier_SingleTest.pdf"
plt.figure(figsize=(fig_width, fig_height))

plt.scatter(fdr_flat, tdr_flat, s=20, color="#cfe8ff", alpha=0.25, zorder=1, rasterized=True)
plt.plot(fdr_only_g[1:], tdr_only_g[1:], color=plt.cm.tab10(4), linewidth=3.5, ls="--", label='only good type')
plt.plot(fdr_only_b[1:], tdr_only_b[1:], color=plt.cm.tab10(2), linewidth=3.5, ls="--", label='only bad type')
plt.plot(fdr_single[1:], tdr_single[1:], color=plt.cm.tab10(1), linewidth=3.5, label='both types')
plt.plot(fdr_list_menu, tdr_list_menu, color=plt.cm.tab10(3), linewidth=3.5, label='Pareto frontier', alpha = 0.8)

plt.xlabel("Maximum FDR", fontsize=fontmed)
plt.ylabel("Total TDR", fontsize=fontmed)
plt.xticks(fontsize=axistick_fontsize)
plt.yticks(fontsize=axistick_fontsize)
plt.title("Single test misses the frontier", fontsize=fontlarge)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc="upper left", fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(fig_directory + figname, format="pdf", bbox_inches="tight")
plt.show()


# ---------Old Code ----------------------

"""
# ----------------- Plot 1: Full FDR/TDR -----------------
figname1 = "TwoAgentFrontier_All.pdf"
plt.figure(figsize=(fig_width, fig_height))
hb = plt.hexbin(fdr_flat, tdr_flat, C=tdr_flat, gridsize=230, cmap="Blues", mincnt=1)

plt.xlabel("Maximum FDR", fontsize=fontmed)
plt.ylabel("Total TDR", fontsize=fontmed)
plt.title("Achievable Tradeoffs", fontsize=fontlarge)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(fig_directory + figname1, format="pdf", bbox_inches="tight")
#plt.show()

# ----------------- Plot 2: Single-test -----------------

figname2 = "TwoAgentFrontier_SingleTest.pdf"
plt.figure(figsize=(fig_width, fig_height))
hb = plt.hexbin(fdr_flat, tdr_flat, C=tdr_flat, gridsize=230, cmap="Blues", mincnt=1)
plt.plot(fdr_only_g[1:], tdr_only_g[1:], color=plt.cm.tab10(4), linewidth=3.5, label='only good type')
plt.plot(fdr_only_b[1:], tdr_only_b[1:], color=plt.cm.tab10(2), linewidth=3.5, label='only bad type')
plt.plot(fdr_single[1:], tdr_single[1:], color=plt.cm.tab10(1), linewidth=3.5, label='both types')

plt.xlabel("Maximum FDR", fontsize=fontmed)
plt.ylabel("Total TDR", fontsize=fontmed)
plt.xticks(fontsize=axistick_fontsize)
plt.yticks(fontsize=axistick_fontsize)
plt.title("Single-Test Tradeoff", fontsize=fontlarge)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc="upper left", fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(fig_directory + figname2, format="pdf", bbox_inches="tight")
#plt.show()

# ----------------- Plot 3: Tailored tests vs Pareto frontier -----------------

figname3 = "TwoAgentFrontier_TailoredTest.pdf"
plt.figure(figsize=(fig_width, fig_height))
hb = plt.hexbin(fdr_flat, tdr_flat, C=tdr_flat, gridsize=230, cmap="Blues", mincnt=1)
plt.plot(fdr_single[1:], tdr_single[1:], color=plt.cm.tab10(1), linewidth=3.5, label='single test')
plt.plot(fdr_list_menu, tdr_list_menu, color=plt.cm.tab10(3), linewidth=3.5, label='Pareto frontier')

plt.xlabel("Maximum FDR", fontsize=fontmed)
plt.ylabel("Total TDR", fontsize=fontmed)
plt.xticks(fontsize=axistick_fontsize)
plt.yticks(fontsize=axistick_fontsize)
plt.title("Gap to Pareto Frontier", fontsize=fontlarge)
plt.grid(True, linestyle='--', alpha=0.5)
#plt.colorbar(hb, label="TDR value")
plt.legend(loc="upper left", fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(fig_directory + figname3, format="pdf", bbox_inches="tight")
#plt.show()"

"""