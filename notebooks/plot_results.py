#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error as mse, mean_absolute_error as mae

def pred_test_curve(y_test, y_pred, y_std=None):
    plt.figure(figsize=(16,7))
    plt.errorbar(y_test, y_pred, yerr=y_std, fmt='x', elinewidth=0.4)
    min_=np.min((y_test,y_pred))
    max_=np.max((y_test,y_pred))
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.plot((min_,max_), (min_,max_), 'r--')
    plt.show()

    
def error_curve(y_test, y_pred, y_std):
    # sort indices from smallest to largest uncertainty
    y_std_sort = np.argsort(y_std)

    # sort indices from smallest to largest squared predictive error
    se_sort = np.argsort(((y_pred - y_test) ** 2))
    err, err_true = [], []
    for j in range(y_test.size, 0, -1):
        less_y_std = y_std_sort[:j]
        less_err = se_sort[:j]
        foo =np.sqrt(mse(y_pred[less_y_std], y_test[less_y_std]))
        err.append(np.sqrt(mse(y_pred[less_y_std], y_test[less_y_std])))
        err_true.append(np.sqrt(mse(y_pred[less_err], y_test[less_err])))

    err=np.array(err)
    err_true=np.array(err_true)    

    plt.figure(figsize=(10,10))
    plt.plot(err, label = "Removing Most Uncertain")
    plt.plot(err_true, label = "Removing Largest Error")
    plt.xlabel('Points Removed')
    plt.ylabel('Error')
    plt.show()  


#%%
import pandas as pd
data = pd.read_csv("/home/reag2/PhD/sampnn/test_results.csv", index_col=0)
y_test, y_pred, y_std = data["target"].values, data["mean"].values, \
                        data["std"].values

print("R2 Score: {:.4f}".format(r2_score(y_test,y_pred)))
print("RMSE:     {:.5f}".format(np.sqrt(mse(y_test,y_pred))))
print("MAE:     {:.5f}".format(mae(y_test,y_pred)))

pred_test_curve(y_test, y_pred, y_std)  
error_curve(y_test, y_pred, y_std)

#%%
