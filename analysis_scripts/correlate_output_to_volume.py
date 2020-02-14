import numpy as np
import os
import pandas as pd

from scipy.stats import spearmanr

from dl_toolbox.data.read import read_baseline_feats

from dl_toolbox.utils import _repetition_dirs, _fold_dirs, _subdirectories_full_path


if __name__ == "__main__":
    # for obtaining the model predictions
    base_dir = "/home/MED/starkeseb/my_experiments/paper_evaluation_of_dl_approaches"
    model_dir = "from_scratch_cox_tanh_no_bn"
    model_cv_dir = os.path.join(base_dir, model_dir)

    # for obtaining the volume data
    data_path_base = "/home/MED/starkeseb/mbro_local/data/DKTK/"
    data_subdirs = _subdirectories_full_path(data_path_base)  # the subcohorts
    baseline_files = [os.path.join(d, "numpy_preprocessed", "clinical_features.csv") for d in data_subdirs]
    vol_col = "ln(GTVtu_from_mask)_zscore"
    baseline_df = read_baseline_feats(baseline_files)
    volume_df = baseline_df[["id", vol_col]]

    print(volume_df)

    train_ids = pd.read_csv("/home/MED/starkeseb/dktk_train_ids.csv", header=None).values.squeeze()

    # correlations among the runs
    spearmans_train = []
    spearmans_test = []
    spearmans_full = []
    pearsons_train = []
    pearsons_test = []
    pearsons_full = []
    exp_dir = []
    for rep_dir in _repetition_dirs(model_cv_dir):
        for fold_dir in _fold_dirs(rep_dir):
            prediction_file = os.path.join(fold_dir, "predictions", "predictions.csv")
            predictions = pd.read_csv(prediction_file)

            pred_col = [c for c in predictions.columns if c.startswith("pred_per_pat")][0]
            pred_df = predictions[["id", pred_col]]

            joined_df = pd.merge(volume_df, pred_df, on="id")
            joined_df_train = joined_df[joined_df.id.isin(train_ids)]
            joined_df_test = joined_df[~joined_df.id.isin(train_ids)]
            # print(joined_df)
            vols_train = joined_df_train[vol_col].values
            pred_train = joined_df_train[pred_col].values
            vols_test = joined_df_test[vol_col].values
            pred_test = joined_df_test[pred_col].values
            vols_full = joined_df[vol_col].values
            pred_full = joined_df[pred_col].values

            # spearman correlation
            spear_train = spearmanr(vols_train, pred_train).correlation
            spearmans_train.append(spear_train)
            spear_test = spearmanr(vols_test, pred_test).correlation
            spearmans_test.append(spear_test)
            spear_full = spearmanr(vols_full, pred_full).correlation
            spearmans_full.append(spear_full)

            # pearson correlation: observations are rows and only a single feature
            # and since this gives the full matrix we take the first nondiagonal element
            pear_train = np.corrcoef(vols_train, pred_train, rowvar=False)[0, 1]
            pearsons_train.append(pear_train)
            pear_test = np.corrcoef(vols_test, pred_test, rowvar=False)[0, 1]
            pearsons_test.append(pear_test)
            pear_full = np.corrcoef(vols_full, pred_full, rowvar=False)[0, 1]
            pearsons_full.append(pear_full)

            exp_dir.append(fold_dir)


    df = pd.DataFrame({
        'experiment': exp_dir,
        'spearmanR_train': spearmans_train,
        'spearmanR_test': spearmans_test,
        'spearmanR_full': spearmans_full,
        'pearsonR_train': pearsons_train,
        'pearsonR_test': pearsons_test,
        'pearsonR_full': pearsons_full
    })

    print(df)
    print(df.describe())




