import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

from sklearn.metrics import roc_auc_score

from ..result_analysis import model_predictions_per_patient
from ..visualization import plot_surv_times_vs_risk, plot_kms


def compute_pval(preds, labels, threshold, alpha):
    """
    preds: np.array (1D) with predictions
    labels. np.array (2D) with n_patients x 2 where first column survival,
    second event status

    Returns
    -------
    p_value of difference between risk groups obtained by using given threshold
    """
    low_risk_idx = np.where(preds <= threshold)[0]
    high_risk_idx = np.where(preds > threshold)[0]

    try:
        test_res = logrank_test(
            labels[low_risk_idx, 0],
            labels[high_risk_idx, 0],
            event_observed_A=labels[low_risk_idx, 1],
            event_observed_B=labels[high_risk_idx, 1],
            alpha=alpha)
        p_val = test_res.p_value
    except Exception as e:
        print("WW: Caught exception in compute_pval", e)
        p_val = np.nan

    return p_val


def compute_ci(preds, labels):
    """
    preds: np.array (1D) with predictions
    labels: np.array (2D) with n_samples rows and columns the survival time
            and censoring information

    Returns
    -------
    concordance index
    """
    times = labels[:, 0]
    events = labels[:, 1]
    try:
        ci = concordance_index(times, preds, events)
    except:
        ci = np.nan

    return ci


def compute_auc(preds, labels):
    """
    preds: np.array(1D) with predictions from a softmax/sigmoid output
    labels: np.array(1D) the class labels
    """

    return roc_auc_score(labels, preds)


def auc_slice(preds, outcomes, ids, id_col="id"):
    preds, labels, _, _ = preds_and_labels_per_slice_and_pat(
        preds, outcomes, ids, id_col=id_col)

    return compute_auc(preds, labels)


def auc_patient(preds, outcomes, ids, id_col="id"):
    _, _, preds, labels = preds_and_labels_per_slice_and_pat(
        preds, outcomes, ids, id_col=id_col)

    return compute_auc(preds, labels)


def ci_slice(preds, outcomes, ids, id_col="id"):
    """
    Parameters
    ----------
    preds: pd.DataFrame
        contains a row for each patient with its id and the slice predictions
        (see dl_toolbox.result_analysis.model_predictions_per_patient)
    outcomes: dict
        mapping patient id to two values (event_time, event_status)
    ids: list/array of patient ids for which the concordance index should be computed
    """
    preds, labels, _, _ = preds_and_labels_per_slice_and_pat(
        preds, outcomes, ids, id_col=id_col)

    return compute_ci(preds, labels)


def ci_patient(preds, outcomes, ids, id_col="id"):

    _, _, preds, labels = preds_and_labels_per_slice_and_pat(
        preds, outcomes, ids, id_col=id_col)

    return compute_ci(preds, labels)


def pval_slice(pred_dict, outcomes, ids, threshold, alpha=0.95, id_col="id"):

    preds, labels, _, _ = preds_and_labels_per_slice_and_pat(
        pred_dict, outcomes, ids, id_col=id_col)

    return compute_pval(preds, labels, threshold, alpha)


def pval_patient(pred_dict, outcomes, ids, threshold, alpha=0.95, id_col="id"):

    _, _, preds, labels = preds_and_labels_per_slice_and_pat(
        pred_dict, outcomes, ids, id_col=id_col)

    return compute_pval(preds, labels, threshold, alpha)


def compute_cis(preds, outcomes, print_result=True, id_col="id"):
    """
    Parameters
    ----------
    preds: pd.DataFrame
        contains a row for each patient with its id and the slice predictions
        (see dl_toolbox.result_analysis.model_predictions_per_patient)
    outcomes: dict
        mapping patient id to two values (event_time, event_status)
    """
    # compute CI scores for training, validation and test
    # (per slice and per patient)

    train, valid, test = _get_preds_and_labels_per_slice_and_patient(
            preds, outcomes, id_col=id_col)

    ret_val = {}
    for cohort, data in zip(["training", "validation", "test"], [train, valid, test]):
        # check if we had data for that cohort
        if data[0] is not None:
            # then all 4 elements are not none
            p_slice, l_slice, p_pat, l_pat = data

            ci_slice = compute_ci(p_slice, l_slice)
            ci_pat = compute_ci(p_pat, l_pat)
        else:
            ci_slice = None
            ci_pat = None
        ret_val[cohort] = (ci_slice, ci_pat)

    cis = {
        'train_ci_slice': ret_val["training"][0],
        'valid_ci_slice': ret_val["validation"][0],
        'test_ci_slice': ret_val["test"][0],

        'train_ci_pat': ret_val["training"][1],
        'valid_ci_pat': ret_val["validation"][1],
        'test_ci_pat': ret_val["test"][1]
    }

    if print_result:
        print("PER SLICE CI: train: {}, valid: {}, test: {}".format(
            cis['train_ci_slice'],
            cis['valid_ci_slice'],
            cis['test_ci_slice']))
        print("PER PATIENT CI: train: {}, valid: {}, test: {}\n".format(
            cis['train_ci_pat'],
            cis['valid_ci_pat'],
            cis['test_ci_pat']))

    return cis


def compute_pvals(preds, outcomes, alpha=0.95,
                  print_result=True, id_col="id"):
    train, valid, test = _get_preds_and_labels_per_slice_and_patient(
            preds, outcomes, id_col=id_col)

    assert train[0] is not None and train[2] is not None
    tr_p_slice, _, tr_p_pat, _ = train
    train_median_risk_slice = np.median(tr_p_slice)
    train_median_risk_pat = np.median(tr_p_pat)

    ret_val = {}
    for cohort, data in zip(["training", "validation", "test"], [train, valid, test]):
        # check if we had data for that cohort
        if data[0] is not None:
            # then all 4 elements are not none
            p_slice, l_slice, p_pat, l_pat = data

            # per slice
            pval_slice = compute_pval(p_slice, l_slice,
                                      threshold=train_median_risk_slice,
                                      alpha=alpha)
            # per patient
            pval_pat = compute_pval(p_pat, l_pat,
                                    threshold=train_median_risk_pat,
                                    alpha=alpha)
        else:
            pval_slice = None
            pval_pat = None

        ret_val[cohort] = (pval_slice, pval_pat)

    pvals = {
        'train_p_slice': ret_val["training"][0],
        'valid_p_slice': ret_val["validation"][0],
        'test_p_slice': ret_val["test"][0],

        'train_p_pat': ret_val["training"][1],
        'valid_p_pat': ret_val["validation"][1],
        'test_p_pat': ret_val["test"][1]
    }

    if print_result:
        print("PER SLICE pvalues: train: {}, valid: {}, test: {}".format(
            pvals['train_p_slice'],
            pvals['valid_p_slice'],
            pvals['test_p_slice']))

        print("PER PATIENT pvalues: train: {}, valid: {}, test: {}\n".format(
            pvals['train_p_pat'],
            pvals['valid_p_pat'],
            pvals['test_p_pat']))

    return pvals


def preds_and_labels_per_slice_and_pat(preds, outcomes, ids,
                                       id_col, aggregate_col=None):
    """
    Parameters
    ----------
    preds: pd.DataFrame
        This requires the columns 'id', 'pred_per_slice' and a column
        that starts with 'pred_per_pat' to be present
    """
    preds_slice = []
    labels_slice = []

    preds_pat = []
    labels_pat = []

    if aggregate_col is None:
        aggregate_col = [c for c in preds.columns.values
                         if c.startswith("pred_per_pat")][0]
        # print("[I]: preds_and_labels_per_slice_and_pat: No 'aggregate_col'"
        #       " was specified, will use", aggregate_col)

    avail_ids = preds[id_col].values
    for id in ids:
        if id not in avail_ids:
            raise ValueError("{} not present in preds!".format(id))
        if id not in outcomes:
            raise ValueError("{} not present in outcomes!".format(id))
        pat_df = preds[preds[id_col] == id]

        lab = outcomes[id]
        risks_per_slice = pat_df["pred_per_slice"].values[0]
        risk_aggregate = pat_df[aggregate_col].values[0]

        # predictions per patient are easy since only one value per id
        preds_pat.append(risk_aggregate)
        labels_pat.append(lab)

        # now for slices we have to loop over them and concatenate the labels
        # for each slice
        for risk in risks_per_slice:
            preds_slice.append(risk)
            labels_slice.append(lab)

    return (np.array(preds_slice).squeeze(), np.array(labels_slice),
            np.array(preds_pat), np.array(labels_pat))


def _get_preds_and_labels_per_slice_and_patient(pred_df, outcomes,
                                                id_col):
    """
    Returns
    -------
    3 tuples of length 4 (for training, validation and test)
    if ids_test is None, the last tuple consist of None values only.

    Each tuple contains 4 numpy arrays, the first being the predictions on a per slice
    basis, the second the survival labels based on a per slice basis, the third the
    predictions aggregated on a per patient basis and the last the survival labels
    on a per patient basis.
    """
    # mapping name of the cohort to the patient ids
    id_map = {}
    if "cohort" in pred_df.columns:
        cohorts = np.unique(pred_df["cohort"].values)
        for c in cohorts:
            id_map[c] = pred_df.loc[pred_df.cohort == c, id_col].values
    else:
        # if no cohorts are given, we assume those all belong to the training
        id_map["training"] = pred_df[id_col].values

    ret_val = {}
    for cohort, ids in id_map.items():
        # pred_per_slice, label_per_slice, pred_per_pat, label_per_pat
        ret_val[cohort] = preds_and_labels_per_slice_and_pat(
            pred_df, outcomes, ids, id_col)

    default = (None,) * 4  # if cohort not present
    train = ret_val.get("training", default)
    valid = ret_val.get("validation", default)
    test = ret_val.get("test", default)

    return train, valid, test


def plot_km_and_scatter(pred_df, outcomes, output_dir, time_in_month=True,
                        id_col="id"):

    train, valid, test = _get_preds_and_labels_per_slice_and_patient(
            pred_df, outcomes, id_col=id_col)

    tr_p_slice, tr_l_slice, tr_p_pat, tr_l_pat = train
    v_p_slice, v_l_slice, v_p_pat, v_l_pat = valid
    te_p_slice, te_l_slice, te_p_pat, te_l_pat = test

    plot_surv_times_vs_risk(
        tr_p_slice, tr_l_slice, v_p_slice, v_l_slice, te_p_slice, te_l_slice,
        subtitle="(per slice)",
        save_dir=os.path.join(output_dir, "risk_vs_survival_per_slice.png"),
        time_in_month=time_in_month)
    plt.close()

    plot_surv_times_vs_risk(
        tr_p_pat, tr_l_pat, v_p_pat, v_l_pat, te_p_pat, te_l_pat,
        subtitle="(per patient)",
        save_dir=os.path.join(output_dir, "risk_vs_survival_per_patient.png"),
        time_in_month=time_in_month)
    plt.close()

    plot_kms(
        tr_p_slice, tr_l_slice, v_p_slice, v_l_slice, te_p_slice, te_l_slice,
        subtitle="(per slice)",
        save_dir=os.path.join(output_dir, "kaplan_meier_per_slice.png"),
        time_in_month=time_in_month)
    plt.close()

    plot_kms(
        tr_p_pat, tr_l_pat, v_p_pat, v_l_pat, te_p_pat, te_l_pat,
        subtitle="(per patient)",
        save_dir=os.path.join(output_dir, "kaplan_meier_per_patient.png"),
        time_in_month=time_in_month)
    plt.close()
