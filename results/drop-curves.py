#%%
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from dcor import distance_correlation as dcorr

plt.rcParams.update({'font.size': 20})

#%%
# Define Helper Functions

def drop_curve(err, y_std):
    # calculate the se for each point
    mse = np.mean(err)

    # sort indices from smallest to largest uncertainty
    err_sort = np.argsort(y_std).tolist()

    # sort indices from smallest to largest squared predictive error
    se_sort = np.argsort(err).tolist()

    res_curve, err_curve = [], []
    res_curve.append(mse)
    err_curve.append(mse)

    for i in range(err.size, 1, -1):
        res_curve.append(res_curve[-1]-(err[se_sort.pop()]-res_curve[-1])/(i-1))
        err_curve.append(err_curve[-1]-(err[err_sort.pop()]-err_curve[-1])/(i-1))

    return res_curve, err_curve

#%%
# Load the Datasets
oqmd = pd.read_csv("results/ensemble_results_f-1_s-0_t-1.csv")
mp = pd.read_csv("results/ensemble_results_f-2_s-0_t-1.csv")
expt = pd.read_csv("results/ensemble_results_f-3_s-0_t-1.csv")
xfer = pd.read_csv("results/ensemble_results_f-4_s-0_t-1.csv")
supcon = pd.read_csv("results/ensemble_results_f-5_s-0_t-1.csv")

# df_list = [oqmd, mp, expt, xfer]
df_list = [oqmd, mp, supcon, expt]
# title_list = ["OQMD Formation Enthalpy Per Atom",
#               "MP Non-Metal Bandgaps",
#               "Expt Non-Metal Bandgaps",
#               "Transfer Non-Metal Bandgaps"]

# title_list = ["OQMD",
#               "MP",
#               "EX",
#               "MP \ then \ EX"]

title_list = ["OQMD",
              "MP",
              "SC",
              "EX"]

#%%
# Plot the curves
fig, ax = plt.subplots(2, 2, figsize=(17,9))
plt.subplots_adjust(top=0.99, bottom=0.1, wspace=0.25, hspace=0.25)

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

    _, ale_curve = drop_curve(res, ale)
    _, epi_curve = drop_curve(res, epi)
    res_curve, std_curve = drop_curve(res, both)

    confidence = np.linspace(0.,100., len(res_curve))

    if i == 0:
        ax[j,k].plot(confidence, std_curve, label="Total Uncertainty")
        ax[j,k].plot(confidence, epi_curve, linestyle=(0, (5, 10)), label="Epistemic Uncertainty")
        ax[j,k].plot(confidence, ale_curve, linestyle=(0, (1, 10)), label="Aleatoric Uncertainty")
        # ax[j,k].plot(confidence, res_curve, label="Ground Truth")
        # ax[j,k].set_title(title, pad=20)
    else:
        ax[j,k].plot(confidence, std_curve)
        ax[j,k].plot(confidence, epi_curve, linestyle=(0, (5, 10)))
        ax[j,k].plot(confidence, ale_curve, linestyle=(0, (1, 10)))
        # ax[j,k].plot(confidence, res_curve)
        
    ax[j,k].legend(title=r"$\bf{Data \ Set: {%s}}$"%(title), frameon=False)
    ax[j,k].set_xlabel("Confidence Percentile", labelpad=10)
    if i == 2:
        ax[j,k].set_ylabel("MAE / K", labelpad=10)
    else:
        ax[j,k].set_ylabel("MAE / eV", labelpad=10)
    ax[j,k].set_xlim((0, 100))
    ax[j,k].set_ylim((0, ax[j,k].get_ylim()[-1]))
        

plt.savefig("drop-curves.pdf")
plt.show()

#%%

# tar = df["target"].to_numpy()

# pred_cols = [col for col in df.columns if 'pred' in col]
# ale_cols = [col for col in df.columns if 'aleatoric' in col]
# pred = df[pred_cols].to_numpy().T
# ales = df[ale_cols].to_numpy().T
# ivar_ale = 1./np.square(ales)
# wmean, wsum = np.average(pred, axis=0,
#                         weights=ivar_ale,
#                         returned=True)

# wale = 1./wsum/len(ale_cols)

# res = np.abs(tar-wmean)

# wepi = np.average((pred-wmean)**2, axis=0,
#                     weights=ivar_ale)

# wboth = wale + wepi
