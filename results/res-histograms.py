#%%
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

#%%
# Load the Datasets
oqmd = pd.read_csv("results/ensemble_results_f-1_s-0_t-1.csv")
mp = pd.read_csv("results/ensemble_results_f-2_s-0_t-1.csv")
expt = pd.read_csv("results/ensemble_results_f-3_s-0_t-1.csv")
xfer = pd.read_csv("results/ensemble_results_f-4_s-0_t-1.csv")
# xfer = pd.read_csv("results/ensemble_results_f-5_s-0_t-1.csv")

df_list = [oqmd, mp, expt, xfer]

title_list = ["OQMD",
              "MP",
              "EX",
              "MP \ then \ EX"]

#%%
# Plot the curves
fig, ax = plt.subplots(2, 2, figsize=(18,13))
plt.subplots_adjust(hspace=0.25, wspace=0.25)

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

    res = np.abs(mean-tar)

    bins = np.linspace(0, np.max([res, both]), 10)

    hists = [both, epi, ale, res]
    labels = ["Total Uncertainty", "Epistemic Uncertainty",
              "Aleatoric Uncertainty", "Ground Truth"]

    if i == 0:
        ax[j,k].hist(hists, bins, alpha=0.5, label=labels)
        # ax[j,k].hist(both, bins, alpha=0.5, label="Total Uncertainty")
        # ax[j,k].hist(epi, bins, alpha=0.5, label="Epistemic Uncertainty")
        # ax[j,k].hist(ale, bins, alpha=0.5, label="Aleatoric Uncertainty")
        # ax[j,k].hist(res, bins, alpha=0.5, label="Ground Truth")
        # ax[j,k].set_title(title, pad=20)
    else:
        ax[j,k].hist(hists, bins, alpha=0.5,)
        # ax[j,k].hist(both, bins, alpha=0.5,)
        # ax[j,k].hist(epi, bins, alpha=0.5,)
        # ax[j,k].hist(ale, bins, alpha=0.5,)
        # ax[j,k].hist(res, bins, alpha=0.5,)
        
    ax[j,k].legend(title=r"$\bf{Data Set: {%s}}$"%(title), frameon=False)
    ax[j,k].set_xlabel("Absolute Error / eV", labelpad=10)
    ax[j,k].set_ylabel("Frequency", labelpad=10)
    ax[j,k].set_yscale('log')
    ax[j,k].set_xlim((0, ax[j,k].get_xlim()[-1]))
    # ax[j,k].set_ylim((0, ax[j,k].get_ylim()[-1]))
        

# plt.savefig("res-hist.pdf")
plt.show()



#%%
