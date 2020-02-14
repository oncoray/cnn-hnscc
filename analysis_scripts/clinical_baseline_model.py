from dl_toolbox.baseline_estimators import CoxModel
from dl_toolbox.preprocessing.stratification import repeated_stratified_cv_splits, exclude_duplicate_dktk_ids
from dl_toolbox.data.read import read_outcome, read_baseline_feats
from dl_toolbox.visualization.kaplan_meier import plot_kms

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os



# read the outcome information and store it as pd.DataFrame
id_col = "ID_Radiomics"
time_col = "LRCtime"
event_col = "LRC"

base_dir = "/home/MED/starkeseb/mbro_local/data/DKTK"
outcome_file = os.path.join(base_dir, "outcome.csv")


outcome_dict = read_outcome(
    outcome_file, id_col=id_col, time_col=time_col, event_col=event_col,
    dropna=True, csv_sep=";")

print(len(outcome_dict))

all_ids = sorted(exclude_duplicate_dktk_ids(
    np.array(list(outcome_dict.keys()))))

train_ids = pd.read_csv("/home/MED/starkeseb/dktk_train_ids.csv", header=None).values.flatten()
print(len(train_ids), len(all_ids))

outcome_df = pd.DataFrame({
    'id': all_ids,
    time_col: [outcome_dict[id][0] for id in all_ids],
    event_col: [outcome_dict[id][1] for id in all_ids]
})

# print(outcome_df)



# read the baseline features (the z-score normalised versions of the volume)
subcohorts = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
# print(subcohorts)

zscore_vol_files = [os.path.join(base_dir, sub, "numpy_preprocessed", "clinical_features.csv")
                       for sub in subcohorts]

zscore_vol_df = read_baseline_feats(zscore_vol_files)
zscore_vol_df = zscore_vol_df.sort_values(by="id")
# print(zscore_vol_df)

# now join the features and the outcomes
baseline_df = pd.merge(zscore_vol_df, outcome_df, on="id")
# print(baseline_df)


# read the unnormalised volumes as well
clinical_feature_files = [os.path.join(base_dir, sub, "numpy_preprocessed", "baseline_features.csv")
                          for sub in subcohorts]

clinical_features_df = read_baseline_feats(clinical_feature_files)
clinical_features_df = clinical_features_df.sort_values(by="id")
clinical_features_df = clinical_features_df[["id", "volume"]]
clinical_features_df.rename({'volume': 'GTVtu_from_mask'}, inplace=True, axis=1)
# print(clinical_features_df)

# also join those information
baseline_df = pd.merge(baseline_df, clinical_features_df, on="id")
print(baseline_df)

# split into training/test

baseline_df_train = baseline_df[baseline_df.id.isin(train_ids)]
baseline_df_test = baseline_df[~baseline_df.id.isin(train_ids)]

print(baseline_df_train.shape, baseline_df_test.shape)

## train the model
feat_cols = ["GTVtu_from_mask"]
# feat_cols = ["ln(GTVtu_from_mask)_zscore"]

model = CoxModel(feat_cols=feat_cols,
                 time_col=time_col, id_col="id", event_col=event_col)

model.fit(baseline_df_train)
train_pred, train_perf = model.predict(baseline_df_train)
# print(train_pred)
print("\nTraining performance:\n======\n", train_perf)

test_pred, test_perf = model.predict(baseline_df_test)
# print(test_pred)
print("\nTest performance:\n======\n", test_perf)

print("\n\nTrain pred\n", train_pred)
# create KM plot
train_labels = baseline_df_train[[time_col, event_col]].values
pred_train = train_pred["pred_per_pat(log_hazard)"].values
test_labels = baseline_df_test[[time_col, event_col]].values
pred_test = test_pred["pred_per_pat(log_hazard)"].values
print(train_labels.shape, pred_train.shape)
print(test_labels.shape, pred_test.shape)


axs, p_vals = plot_kms(
    pred_train=pred_train, train_labels=train_labels,
    pred_valid=pred_test, valid_labels=test_labels,
    titles=["Exploration", "Test"],
    table_below_scaling=0.1,  # larger number -> table far from plot
)


f = axs[0].figure

output_dir = "/home/MED/starkeseb/my_experiments/paper_evaluation_of_dl_approaches/"
output_file = os.path.join(output_dir, f"clinical_model_kaplan_meier_{feat_cols[0]}.png")
f.savefig(output_file)
print("stored KM plot to", output_file)

# # ## Performing cross validation

# # In[24]:





# # In[51]:


# test_df = baseline_df_test

# feat_cols = [
#     "volume",
#     #"Age"
# ]
# ci_col = "CI_hazard" # could also use "CI_time"

# folds = repeated_stratified_cv_splits(
#     ids=train_ids, events=[outcome_dict[pat][1] for pat in train_ids],
#     cv_k=10, cv_reps=50)

# #print(len(folds), len(folds[0]), len(folds[0][0]), len(folds[0][0][0]), len(folds[0][0][1]))
# train_cis = []
# valid_cis = []
# test_cis = []
# for rep in range(len(folds)):
#     for fold in range(len(folds[rep])):
#         print("rep={}/{}, fold={}/{}".format(rep+1, len(folds), fold+1, len(folds[rep])))

#         train_ids, valid_ids = folds[rep][fold]

#         train_df = baseline_df_train[baseline_df_train.id.isin(train_ids)]
#         valid_df = baseline_df_train[baseline_df_train.id.isin(valid_ids)]

#         model = CoxModel(feat_cols=feat_cols,
#                  time_col=time_cols["DKTK"], id_col="id", event_col=event_cols["DKTK"],
#                  verbose=False)

#         model.fit(train_df)

#         pred_train, perf_train = model.predict(train_df)
#         pred_valid, perf_valid = model.predict(valid_df)
#         pred_test, perf_test = model.predict(test_df)


#         ci_train = perf_train[ci_col].values[0]
#         ci_valid = perf_valid[ci_col].values[0]
#         ci_test = perf_test[ci_col].values[0]

#         print("CI train={}, CI valid={}, CI test={}".format(
#             ci_train, ci_valid, ci_test))

#         train_cis.append(ci_train)
#         valid_cis.append(ci_valid)
#         test_cis.append(ci_test)


# # In[52]:


# crossval_ci_df = pd.DataFrame({'train_cis': train_cis, 'valid_cis': valid_cis, 'test_cis': test_cis})
# crossval_ci_df.describe()


# # In[ ]:




