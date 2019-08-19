#%%
# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse, \
                            mean_absolute_error as mae, \
                            r2_score
import pandas as pd

from scipy.stats import spearmanr, pearsonr
from dcor import distance_correlation as dcorr

#%%
# Define internal functions


def pred_test_curve(y_test, y_pred, y_std=None):
    plt.figure(figsize=(10, 6))
    plt.errorbar(y_test, y_pred, yerr=y_std, fmt='x', elinewidth=0.4)
    min_ = np.min((y_test, y_pred))
    max_ = np.max((y_test, y_pred))
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.plot((min_, max_), (min_, max_), 'r--')
    plt.show()


def error_curve_orig(y_test, y_pred, y_std):
    # sort indices from smallest to largest uncertainty
    y_std_sort = np.argsort(y_std)

    # sort indices from smallest to largest squared predictive error
    se_sort = np.argsort(((y_pred - y_test) ** 2))
    err, err_true = [], []
    for j in range(y_test.size, 0, -1):
        less_y_std = y_std_sort[:j]
        less_err = se_sort[:j]
        err.append(np.sqrt(mse(y_pred[less_y_std], y_test[less_y_std])))
        err_true.append(np.sqrt(mse(y_pred[less_err], y_test[less_err])))

    err = np.array(err)
    err_true = np.array(err_true)

    plt.figure(figsize=(10, 6))
    plt.plot(err, label="Removing Most Uncertain")
    plt.plot(err_true, label="Removing Largest Error")
    plt.xlabel('Points Removed')
    plt.ylabel('Error')
    plt.show()

    return err

def error_curve(y_test, y_pred, y_std):
    # calculate the se for each point
    se = np.square(y_pred - y_test)
    mse = np.mean(se)

    # sort indices from smallest to largest uncertainty
    err_sort = np.argsort(y_std).tolist()

    # sort indices from smallest to largest squared predictive error
    se_sort = np.argsort(se).tolist()

    res_curve, err_curve = [], []
    res_curve.append(mse)
    err_curve.append(mse)

    for i in range(y_test.size, 1, -1):
        res_curve.append(res_curve[-1]-(se[se_sort.pop()]-res_curve[-1])/(i-1))
        err_curve.append(err_curve[-1]-(se[err_sort.pop()]-err_curve[-1])/(i-1))

    res_curve = np.sqrt(res_curve)
    err_curve = np.sqrt(err_curve)

    plt.figure(figsize=(10,6))
    plt.plot(err_curve, label = "Removing Most Uncertain")
    plt.plot(res_curve, label = "Removing Largest Error")
    plt.xlabel('Points Removed')
    plt.ylabel('Error')
    plt.show()

    return err_curve

#%%
# Load results
data = pd.read_csv("/home/reag2/PhD/first-year/sampnn/results/ensemble_results_f-2_s-0_t-1.csv", index_col=0)
y_test, y_pred, y_std = data["target"].values, data["mean"].values, \
                        data["std"].values

#%%
# Calculate metrics
print("R2 Score: {:.4f}".format(r2_score(y_test, y_pred)))
print("RMSE:     {:.5f}".format(np.sqrt(mse(y_test, y_pred))))
print("MAE:     {:.5f}".format(mae(y_test, y_pred)))

rs, _ = spearmanr(np.abs(y_test-y_pred), y_std)
rp, _ = pearsonr(np.abs(y_test-y_pred), y_std)
rd = dcorr(np.abs(y_test-y_pred), y_std)

print("Spearman:    {:.4f}".format(rs))
print("Pearson:     {:.4f}".format(rp))
print("Distance:    {:.4f}".format(rd))



#%%
# Plot Results
# pred_test_curve(y_test, y_pred, y_std)
a = error_curve(y_test, y_pred, y_std)
b = error_curve_orig(y_test, y_pred, y_std)



#%%
plt.plot(a, label="test")
plt.plot(b, '--', label="world")
plt.legend()
plt.show()

#%%
np.logspace(0, 2.7, 9, dtype=int)[::-1]

#%%
