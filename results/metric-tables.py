
#%%
# Import Libraries
import pandas as pd
import numpy as np
import copy
from scipy.stats import spearmanr, pearsonr, kendalltau
from dcor import distance_correlation

#%%
# Define Helper Functions


def distancer(X, Y, n_ptest=1000):
    """
    calculate the distance correlation and estimate it's p-value
    """

    rd = distance_correlation(X, Y)

    greater = 0
    for _ in range(n_ptest):
        _Y = copy.copy(Y)
        np.random.shuffle(_Y)
        if distance_correlation(X, _Y) > rd:
            greater += 1

    return rd, greater / float(n_ptest)


def error_metrics(y_test, y_pred, verbose=False):
    """
    calculate a variety of error metrics for regression tasks
    """

    r2 = r2_score(y_test, y_pred)

    res = np.abs(y_test - y_pred)
    mae_avg = np.mean(res)
    mae_std = np.std(res)/np.sqrt(len(res))

    se = np.square(y_test - y_pred)
    mse_avg = np.mean(se)
    mse_std = np.std(se)/np.sqrt(len(se))

    rmse_avg = np.sqrt(mse_avg)
    rmse_std = 0.5 * rmse_avg * mse_std / mse_avg

    if verbose:
        print("R2 Score:    {:.4f} ".format(r2))
        print("MAE:         {:.4f} +/- {:.4f}".format(mae_avg, mae_std))
        print("RMSE:        {:.4f} +/- {:.4f}".format(rmse_avg, rmse_std))

    return {"R2":r2,
            "MAE":(mae_avg, mae_std),
            "RMSE":(rmse_avg, rmse_std)}


def residual_correlations(y_test, y_pred, y_std, verbose=False):
    """
    calculate a variety of correlation scores between the residuals
    and the estimated standard deviations. 
    """
    
    res = np.abs(y_test-y_pred)

    rs, ps = spearmanr(res, y_std)
    rp, pp = pearsonr(res, y_std)
    kt, pt = kendalltau(res, y_std)
    rd, pd = distancer(res, y_std)

    if verbose:
        print("Spearman:    {:.4f}      p-value: {:.4f}".format(rs, ps))
        print("Pearson:     {:.4f}      p-value: {:.4f}".format(rp, pp))
        print("Kendall Tau: {:.4f}      p-value: {:.4f}".format(kt, pt))
        print("Distance:    {:.4f}      p-value: {:.4f}".format(rd, pd))

    return {"Spearman":(rs, ps),
            "Pearson":(rp, pp),
            "Kendall":(kt,pt),
            "Distance":(rd,pd)}

#%%
# Evaluate metrics MPNN
df = pd.read_csv("results/ensemble_results_f-4_s-0_t-1.csv".format(i))

tar = df["target"].to_numpy()

pred_cols = [col for col in df.columns if 'pred' in col]
pred = df[pred_cols].to_numpy().T
mean = np.average(pred, axis=0)

epi = np.var(pred, axis=0)

ale_cols = [col for col in df.columns if 'aleatoric' in col]
ales = df[ale_cols].to_numpy().T
ale = np.mean(np.square(ales), axis=0)

both = epi + ale

_ = error_metrics(tar, mean, verbose=True)
_ = residual_correlations(tar, mean, both, verbose=True)

#%%
# Evaluate Metrics RF

df = pd.read_csv("results/mp_results.csv".format(i))

y_test = df["target"]
y_pred = df["mean"]
y_std = df["std"]

_ = error_metrics(y_test, y_pred, verbose=True)
_ = residual_correlations(y_test, y_pred, y_std, verbose=True)