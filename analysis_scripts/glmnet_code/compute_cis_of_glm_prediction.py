import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from lifelines.utils import concordance_index

from dl_toolbox.result_analysis.evaluate_model import compute_cis, compute_pvals
from dl_toolbox.cross_validation.ensemble import evaluate_ensemble_cindex, get_ensemble_preds_from_cv_results


def create_new_format_df(df, aggregate_fun=np.mean):
    """brings the dataframe into the same shape as our cv pipeline in python has."""
    # get the unique patient ids
    pat_ids = np.unique(df.id.values)

    # for each patient, get the slices
    pat_preds = [None] * len(pat_ids)
    pat_slice_preds = [None] * len(pat_ids)
    pat_time = [None] * len(pat_ids)
    pat_event = [None] * len(pat_ids)
    pat_var = [None] * len(pat_ids)
    pat_cohort = [None] * len(pat_ids)

    for i, pat in enumerate(pat_ids):
        pat_df = df[df.id == pat]

        time = np.unique(pat_df.LRCtime.values)[0]
        event = np.unique(pat_df.LRC.values)[0]
        cohort = np.unique(pat_df.cohort)[0]

        slice_preds = pat_df.glm_prediction.values
        pat_pred = aggregate_fun(slice_preds)
        #print("{} has {} predictions, its aggregate is {} and its label is {}".format(
        #    pat, len(slice_preds), pat_pred, (time, event)))
        pat_cohort[i] = cohort
        pat_slice_preds[i] = slice_preds
        pat_preds[i] = pat_pred
        pat_time[i] = time
        pat_event[i] = event
        pat_var[i] = np.var(slice_preds)

    new_df = pd.DataFrame({
        'id': pat_ids,
        'cohort': pat_cohort,
        'LRCtime': pat_time,
        'LRC': pat_event,
        'pred_per_slice': pat_slice_preds,
        'pred_per_pat': pat_preds,
        'pred_variance': pat_var
    })

    return new_df


# copy paste from the dl_toolbox cv_context
def create_performance_df(pred_df, outcome_dict, r, k):
    cis = compute_cis(
        pred_df, outcome_dict)

    # compute p values for stratification for training, validation and test (per slice and per patient)
    # based on the median risk of the training set
    pvals = compute_pvals(
        pred_df, outcome_dict)

    perf_df = pd.DataFrame({
        'train_ci_slice': cis['train_ci_slice'],
        'p_val_train_slice': pvals['train_p_slice'],

        'train_ci_pat': cis['train_ci_pat'],
        'p_val_train_pat': pvals['train_p_pat'],

        'valid_ci_slice': cis['valid_ci_slice'],
        'p_val_valid_slice': pvals['valid_p_slice'],

        'valid_ci_pat': cis['valid_ci_pat'],
        'p_val_valid_pat': pvals['valid_p_pat'],

        'test_ci_slice': cis['test_ci_slice'],
        'p_val_test_slice': pvals['test_p_slice'],

        'test_ci_pat': cis['test_ci_pat'],
        'p_val_test_pat': pvals['test_p_pat']}, index=[0])

    perf_df.insert(loc=0, column="rep", value=r)
    perf_df.insert(loc=1, column="fold", value=k)

    return perf_df


def create_pred_df(df, r, k):
    # make new format so we can use our dl_toolbox to compute cis
    pred_df = create_new_format_df(df)
    # add rep and fold information
    pred_df.insert(loc=1, column="rep", value=r)
    pred_df.insert(loc=2, column="fold", value=k)

    return pred_df




base_dir_predictions = ("/home/MED/starkeseb/my_experiments/"
    "paper_evaluation_of_dl_approaches/autoencoder/glmnet_performance/")

rep = "rep_0"
fold = "fold_0"
fold_dir = os.path.join(base_dir_predictions, rep, fold)

df_train = pd.read_csv(os.path.join(fold_dir, "glm_pred_train.csv"))
df_valid = pd.read_csv(os.path.join(fold_dir, "glm_pred_valid.csv"))  # need to rename the 'test' cohort entries in 'validation'
df_test = pd.read_csv(os.path.join(fold_dir, "glm_pred_test.csv"))

print(df_train.head(5))
print(df_valid.head(5))
print(df_test.head(5))


df_valid.replace({'test': 'validation'}, inplace=True)

df = pd.concat([df_train, df_valid, df_test], axis=0)
print(df)
print(df["cohort"].value_counts())

print(df_train.shape)
print(df_valid.shape)
print(df_test.shape)
print(df.shape)


df_new = create_new_format_df(df)
print(df_new)

outcomes = {row.id: (row.LRCtime, row.LRC) for _, row in df_new.iterrows()}
print(outcomes)

compute_cis(df_new, outcomes)
compute_pvals(df_new, outcomes)


# ## Now loop over all the folds and repetitions
pred_dict = {}
perf_dict = {}
reps = [d for d in os.listdir(base_dir_predictions) if d.startswith("rep_")]
for rep in reps:
    r = int(rep.split("_")[1])
    rep_dir = os.path.join(base_dir_predictions, rep)

    pred_dict[rep] = {}
    perf_dict[rep] = {}

    folds = [d for d in os.listdir(rep_dir) if d.startswith("fold_")]
    for fold in folds:
        k = int(fold.split("_")[1])
        fold_dir = os.path.join(rep_dir, fold)
        print(fold_dir, "\n")

        df_train = pd.read_csv(os.path.join(fold_dir, "glm_pred_train.csv"))
        # replace the cohort in this
        df_valid = pd.read_csv(os.path.join(fold_dir, "glm_pred_valid.csv"))
        df_valid.replace({'test': 'validation'}, inplace=True)

        df_test = pd.read_csv(os.path.join(fold_dir, "glm_pred_test.csv"))

        # concatenate to single df
        df = pd.concat([df_train, df_valid, df_test], axis=0)
        pred_df = create_pred_df(df, r, k)

        cis = compute_cis(pred_df, outcomes)

        cis_slice = {k: v for k, v in cis.items() if "_slice" in k}
        cis_pat = {k: v for k, v in cis.items() if "_pat" in k}

        ps = compute_pvals(df_new, outcomes)
        ps_slice = {k: v for k, v in ps.items() if "_slice" in k}
        ps_pat = {k: v for k, v in ps.items() if "_pat" in k}

        print("\n{}, {}:\n{}\n{}".format(rep, fold, cis_slice, cis_pat))
        print("{}\n{}".format(ps_slice, ps_pat))


        perf_df = create_performance_df(pred_df, outcomes, r, k)

        pred_dict[rep][fold] = pred_df
        perf_dict[rep][fold] = perf_df

print(perf_dict["rep_0"]["fold_0"])
all_performance_df = pd.concat([perf_dict[rep][fold] for rep in perf_dict for fold in perf_dict[rep]], axis=0)
print(all_performance_df)

print(all_performance_df[[c for c in all_performance_df.columns if "_pat" in c ]].describe())

print("Number of runs with test p-val < 0.05", len(all_performance_df[all_performance_df["p_val_test_pat"] < 0.05]))


# ## Ensemble prediction
out_path = "/home/MED/starkeseb/tmp/lasso_cox_ensemble"
os.makedirs(out_path, exist_ok=True)

ensemble_res = get_ensemble_preds_from_cv_results(
                pred_dict, outcomes,
                aggregate_col=None,
                ensemble_method=np.mean,
                output_dir=out_path)

train_preds_ensemble, train_labels = ensemble_res[0]
valid_preds_ensemble, valid_labels = ensemble_res[1]
test_preds_ensemble, test_labels = ensemble_res[2]
# print(train_preds_ensemble.shape, train_labels.shape)
# print(valid_preds_ensemble.shape, valid_labels.shape)
# print(test_preds_ensemble.shape, test_labels.shape)

ret_val = evaluate_ensemble_cindex(
    train_preds_ensemble, train_labels,
    valid_preds_ensemble, valid_labels,
    test_preds_ensemble, test_labels,
    output_dir=out_path)

print(ret_val)





