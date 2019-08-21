#%%
# evaluate how ensembling improves metrics
from itertools import combinations

ensemble_folds = 10
for i in range(ensemble_folds):
    df = pd.read_csv("results/ensemble_results_f-2_s-0_t-1.csv".format(i))
    
    pred_cols = [col for col in df.columns if 'pred' in col]
    res_list = [df[col].to_numpy()-df["target"].to_numpy() for col in pred_cols]

#%%
# work out metrics for all possible combinations

mae_comb = np.zeros((ensemble_folds))
rmse_comb = np.zeros_like(mae_comb)

for i in range(ensemble_folds):
    count = 0
    for indicies in combinations(range(ensemble_folds), i+1):
        count += 1
        comb = [res_list[j] for j in indicies]
        mae_comb[i] += np.mean(np.abs(np.mean(comb, axis=0)))
        rmse_comb[i] += np.mean(np.square(np.mean(comb, axis=0)))
        print("ding")
        break
    mae_comb[i] /= count
    rmse_comb[i] /= count

print(mae_comb)

#%%

fig, ax = plt.subplots(1, 2, figsize=(18,8))
ax[0].plot(np.arange(1, 11), mae_comb)
ax[1].plot(np.arange(1, 11), mae_comb)
ax[1].set_xscale('log')
ax[1].set_yscale('log')
plt.show()


#%%
