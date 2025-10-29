import matplotlib.pyplot as plt
import utils as NMT
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad, cumulative_trapezoid
from GaussianBinaryTest import *

fig_directory = "./Figure/"

Test = GaussianBinaryTest(mu=1, sigma=1.0)
pi_list = np.array([0.3,0.4,0.5,0.6,0.7])
fdr_level = 0.25
tau_list = np.array([Test.optimal_threshold(pi, fdr_level) for pi in pi_list])
base_reward = 100
base_cost = 5
epsilon = 100 #[50, 100]
cost_position = "Mid"

taus, rewards, costs = Test.build_sep_menu(pi_list, tau_list, base_reward, base_cost, epsilon, cost_position)



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

pi_range = np.linspace(0, 1, 1000)
num_contracts = len(taus)
colors = plt.cm.tab10.colors  # You can pick any colormap you like

# Compute utilities: shape (num_contracts, len(pi_range))
ys = np.array([
    Test.utility(pi_range, taus[i], rewards[i], costs[i])
    for i in range(num_contracts)
])

# Find max index at each point
max_indices = np.argmax(ys, axis=0)  # shape: (len(pi_range),)

# Plot the upper envelope with color-coded segments
for i in range(num_contracts):
    mask = max_indices == i
    if np.any(mask):
        idx = np.where(mask)[0]
        splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for segment in splits:
            x_seg = pi_range[segment]
            y_seg = ys[i][segment]
            plt.plot(x_seg, y_seg, color=colors[i % len(colors)], label=f'contract {i+1}' if segment[0] == idx[0] else None)
            plt.fill_between(x_seg, 0, y_seg, color=colors[i % len(colors)], alpha=0.2)


# Interpolate max utility values and dominating indices at pi_list points
utility_at_pis = []
dominant_colors = []
for pi_val in pi_list:
    idx_closest = np.argmin(np.abs(pi_range - pi_val))
    contract_idx = max_indices[idx_closest]
    utility_val = ys[contract_idx][idx_closest]
    utility_at_pis.append(utility_val)
    dominant_colors.append(colors[contract_idx % len(colors)])
    plt.plot(pi_val, utility_val, 'o', color=colors[contract_idx % len(colors)], markersize=6, label=None)


# Annotate the utility envelope
# Choose a pi value and corresponding utility on the envelope for placement
annotate_idx = 100  # adjust this index for preferred position
annotate_x = pi_range[annotate_idx]
annotate_y = ys[max_indices[annotate_idx]][annotate_idx]

plt.annotate(r"Utility envelope $G(q)$", 
             xy=(annotate_x, annotate_y), 
             xytext=(annotate_x + 0.05, annotate_y + 0.5),  # offset the text position
             arrowprops=dict(arrowstyle='->', color='black', lw=1),
             fontsize=fontmed)


##change x-axis ticks
new_ticks = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
plt.xticks(new_ticks)

    
# Final plot touches
plt.xlabel("Agent type",fontsize = fontmed)
plt.ylabel('Utility',fontsize = fontmed)
plt.title('Utility-maximizing contract by agent type',fontsize = fontlarge)
plt.grid(linestyle = '--', alpha=0.5)
plt.legend(loc = "upper right", fontsize = legend_fontsize)
plt.tick_params(axis='both', labelsize = axistick_fontsize)
plt.tight_layout()

figname = f"DiscreteMenu_eps{epsilon}_cost{cost_position}.pdf"
plt.savefig(fig_directory + figname, format="pdf", bbox_inches="tight")

plt.show()