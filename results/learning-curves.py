#%%
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from sklearn.metrics import mean_squared_error as mse, \
                            mean_absolute_error as mae, r2_score
plt.rcParams.update({'font.size': 20})

sizes = np.logspace(0, 2.7, 9, dtype=int)[::-1]

#%%
# Get learning curve lists for OQMD
oqmd_rmse_aoe = []
oqmd_mae_aoe = []
oqmd_r2_aoe = []

for i in sizes:
    df = pd.read_csv("results/ensemble_results_f-1_s-0_t-{}.csv".format(i))

    y_test = df["target"]
    pred_cols = [col for col in df.columns if 'pred' in col]
    ale_cols = [col for col in df.columns if 'aleatoric' in col]
    y_pred = np.average(df[pred_cols], axis=1)
    # y_pred = np.average(df[pred_cols],
    #                     weights=1./np.square(df[ale_cols]),
    #                     axis=1)


    rmse = np.sqrt(mse(y_test, y_pred))
    oqmd_rmse_aoe.append(rmse)
    
    res = mae(y_test, y_pred)
    oqmd_mae_aoe.append(res)
    
    r2 = r2_score(y_test, y_pred)
    oqmd_r2_aoe.append(r2)


#%%
# Get learning curve lists for MP
mp_rmse_aoe = []
mp_mae_aoe = []
mp_r2_aoe = []

for i in sizes:
    df = pd.read_csv("results/ensemble_results_f-2_s-0_t-{}.csv".format(i))

    y_test = df["target"]
    pred_cols = [col for col in df.columns if 'pred' in col]
    ale_cols = [col for col in df.columns if 'aleatoric' in col]
    y_pred = np.average(df[pred_cols].to_numpy(), axis=1)
    # y_pred = np.average(df[pred_cols].to_numpy(),
    #                     weights=1./np.square(df[ale_cols]),
    #                     axis=1)

    rmse = np.sqrt(mse(y_test, y_pred))
    mp_rmse_aoe.append(rmse)
    
    res = mae(y_test, y_pred)
    mp_mae_aoe.append(res)
    
    r2 = r2_score(y_test, y_pred)
    mp_r2_aoe.append(r2)



#%%
# Load Learning Curves for Baseline

mp_rf = pd.read_csv("results/mp_learn_curve.csv")
mp_size_rf = mp_rf["n_train"]
mp_mae_rf = mp_rf["mae"]
mp_rmse_rf = mp_rf["rmse"]

oqmd_rf = pd.read_csv("results/oqmd_learn_curve.csv")
oqmd_size_rf = oqmd_rf["n_train"]
oqmd_mae_rf = oqmd_rf["mae"]
oqmd_rmse_rf = oqmd_rf["rmse"]

oqmd_elemnet = pd.read_csv("results/elemnet-learn.csv")
oqmd_size_elemnet = oqmd_elemnet["num"]
oqmd_mae_elemnet = oqmd_elemnet["mae"]

#%%
# MAE Learning Curve
fig, ax = plt.subplots(2, figsize=(10,14))
# fig, ax = plt.subplots(1, 2, figsize=(18,6))
plt.subplots_adjust(top=0.95, bottom=0.1, wspace=0.2)

# Materials Project
ax[0].plot(mp_size_rf, mp_mae_rf, label="Baseline Model", marker='x', markersize=12)
ax[0].plot(mp_size_rf.tolist(), mp_mae_aoe, label="Present Work", marker='x', markersize=12)

# ax[0].set_title("MP Non-Metal Bandgaps",  pad=20)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_yticks((1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4))
ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[0].set_xlabel("Number of Training Points", labelpad=10)
ax[0].set_ylabel("MAE / eV", labelpad=10)
ax[0].legend(title=r"$\bf{Data\ Set: MP}$", loc=3, frameon=False)

# Open Quantum Materials Database
ax[1].plot(oqmd_size_rf, oqmd_mae_rf, label="Baseline Model", marker='x', markersize=12)
ax[1].plot(oqmd_size_rf.tolist(), oqmd_mae_aoe, label="Present Work", marker='x', markersize=12)
# ax[1].plot([oqmd_size_rf.tolist()[-1]], [0.05], linestyle='none', label="ElemNet", marker="x", markersize=12)
ax[1].plot(oqmd_size_elemnet, oqmd_mae_elemnet, label="ElemNet", marker="x", markersize=12)

# ax[1].set_title("OQMD Formation Enthalpy Per Atom",  pad=20)
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_yticks((0.3, 0.2, 0.14, 0.1, 0.07, 0.05, 0.03))
ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1].set_xlabel("Number of Training Points", labelpad=10)
ax[1].set_ylabel("MAE / eV ", labelpad=10)
ax[1].legend(title=r"$\bf{Data\ Set: OQMD}$", loc=3, frameon=False)


fig.savefig("learning-curves-mae-vert.pdf", transparent=True)
# fig.savefig("learning-curves-mae.pdf")
fig.show()

#%%
# RMSE Learning Curve
fig, ax = plt.subplots(1, 2, figsize=(18,8))

# Materials Project
ax[0].plot(mp_size_rf, mp_rmse_rf, label="Baseline Model", marker='x', markersize=12)
ax[0].plot(mp_size_rf.tolist(), mp_rmse_aoe, label="Present Work", marker='x', markersize=12)

# # inset plot to show the cross-over for the materials project data
axins = zoomed_inset_axes(ax[0], zoom=5, bbox_to_anchor=(.02, .015, .3, .4),
            bbox_transform=ax[0].transAxes, loc=3)
axins.plot(mp_size_rf, mp_rmse_rf, "-")
axins.plot(mp_size_rf.tolist(), mp_rmse_aoe, "-")

axins.set_xlim(25000,39000) # apply the x-limits
axins.set_ylim(0.67, 0.73) # apply the y-limits

plt.yticks(visible=False)
plt.xticks(visible=False)
mark_inset(ax[0], axins, loc1=1, loc2=4, fc="none", ec="0.5")

# ax[0].set_title("MP Non-Metal Bandgaps",  pad=20)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_yticks((1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5))
ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[0].set_xlabel("Number of Training Points", labelpad=10)
ax[0].set_ylabel("RMSE / eV", labelpad=10)
ax[0].legend(title="MP Non-Metal Bandgaps")


# Open Quantum Materials Database
ax[1].plot(oqmd_size_rf, oqmd_rmse_rf, label="Baseline Model", marker='x', markersize=12)
ax[1].plot(oqmd_size_rf.tolist(), oqmd_rmse_aoe, label="Present Work", marker='x', markersize=12)

# ax[1].set_title("OQMD Formation Enthalpy Per Atom",  pad=20)
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_yticks((0.3, 0.2, 0.14, 0.1))
ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax[1].set_xlabel("Number of Training Points", labelpad=10)
ax[1].set_ylabel("RMSE / eV ", labelpad=10)
ax[1].legend(title="OQMD Formation Enthalpy Per Atom")

fig.savefig("learning-curves-rmse.pdf")
fig.show()



