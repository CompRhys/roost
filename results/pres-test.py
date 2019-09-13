#%%
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
plt.rcParams.update({'font.size': 20})

#%%
# Load the Datasets
oqmd = pd.read_csv("results/ensemble_results_f-1_s-0_t-1.csv")
mp = pd.read_csv("results/ensemble_results_f-2_s-0_t-1.csv")
expt = pd.read_csv("results/ensemble_results_f-3_s-0_t-1.csv")
# xfer = pd.read_csv("results/ensemble_results_f-4_s-0_t-1.csv")
supcon = pd.read_csv("results/ensemble_results_f-5_s-0_t-1.csv")

df_list = [oqmd, mp, supcon, expt]

title_list = ["OQMD",
              "MP",
              "SC",
              "EX"]

#%%
# Plot the curves
fig, ax = plt.subplots(2, 2, figsize=(17,9))
plt.subplots_adjust(top=0.99, bottom=0.1, wspace=0.20, hspace=0.25,
                    left=0.08, right=0.95,)

for i, (df, title) in enumerate(zip(df_list, title_list)):
    j, k = divmod(i, 2)

    tar = df["target"].to_numpy()

    pred_cols = [col for col in df.columns if 'pred' in col]
    pred = df[pred_cols].to_numpy().T
    mean = np.average(pred, axis=0)

    epi = np.var(pred, axis=0)

    ale_cols = [col for col in df.columns if 'aleatoric' in col]
    ales = df[ale_cols].to_numpy().T
    ale = np.mean(np.square(ales), axis=0)
    
    both = epi + ale

    r2 = r2_score(tar, mean)

    im = ax[j,k].scatter(tar, mean, c=both, facecolors='none',
                            norm=matplotlib.colors.LogNorm())
        
    ax[j,k].legend(title=r"$\bf{Data \ Set: {%s}$"%(title), frameon=False)
    if i == 2:
        ax[j,k].set_xlabel("Target Value / K", labelpad=10)
        ax[j,k].set_ylabel("Predicted Value / K", labelpad=8)
    else:
        ax[j,k].set_xlabel("Target Value / eV", labelpad=10)
        ax[j,k].set_ylabel("Predicted Value / eV", labelpad=8)
    # ax[j,k].set_xlim((0, ax[j,k].get_xlim()[-1]))
    # ax[j,k].set_ylim((0, ax[j,k].get_ylim()[-1]))
    cbar = fig.colorbar(im, ax=ax[j,k])

    cbar.ax.get_yaxis().labelpad = 25
    cbar.ax.set_ylabel("Uncertainty", rotation=270)
        

plt.savefig("pred-test.pdf")
plt.show()



#%%
