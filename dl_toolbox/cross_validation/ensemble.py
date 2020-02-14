import numpy as np
import os
import pandas as pd

from ..result_analysis.evaluate_model import compute_ci
from ..visualization import plot_kms, plot_surv_times_vs_risk, plot_auc_curves,\
    plot_confusion_matrices

from sklearn.metrics import roc_auc_score


def df_for_each_patient(predictions, id_col="id"):
    """
    Creates a dataframe for each patient where columns are
    rep|fold|cohort|pred_per_slice|... and stores them in a
    dictionary.
    """

    dfs = []
    for rep in predictions:
        # r = int(rep.split("_")[1])
        for fold in predictions[rep]:
            # k = int(fold.split("_")[1])
            pred_df = predictions[rep][fold]
            dfs.append(pred_df)

    df_concat = pd.concat(dfs, ignore_index=True, sort=False)
    ids = np.unique(df_concat[id_col].values)
    patient_df = {}
    for id in ids:
        patient_df[id] = df_concat[df_concat[id_col] == id]

    return patient_df


def predictions_by_cohort_for_single_patient(df, pred_col="pred_per_slice"):
    """
    Parameters
    ----------
    df: pd.DataFrame
        as returned by 'df_for_each_patient' method with results
        for all repetitions and folds of cross validation for a single
        patient.
    Returns
    -------
    three tuples of length two (for training, validation and test).
    The first element of each tuple contains a list of strings that combine
    the repetition number and the fold number in the form "rep_{}_fold_{}"
    for all the cv runs that the patient belonged to the respective cohort (training, validation or test).

    The second element of each tuple is a numpy array of the same length as the first element
    and containing the predictions given by 'pred_col'
    argument for the respective simulation runs.

    Depending on whether the patient belonged to the training cohort used in cross-validation
    or to the holdout set, either the first two tuples or the last tuple might contain empty
    lists/arrays since the patient was never part of the training/test cohort.
    """
    in_training = df[df.cohort == "training"]
    runs_in_training = [
        "rep_" + str(r) + "_fold_" + str(k)
        for (r, k) in zip(in_training.rep, in_training.fold)]
    preds_in_training = in_training[pred_col].values
    preds_in_training = np.array([preds for preds in preds_in_training])

    in_validation = df[df.cohort == "validation"]
    runs_in_validation = [
        "rep_" + str(r) + "_fold_" + str(k)
        for (r, k) in zip(in_validation.rep, in_validation.fold)]
    preds_in_validation = in_validation[pred_col].values
    preds_in_validation = np.array([preds for preds in preds_in_validation])

    in_test = df[df.cohort == "test"]
    runs_in_test = [
        "rep_" + str(r) + "_fold_" + str(k)
        for (r, k) in zip(in_test.rep, in_test.fold)]
    preds_in_test = in_test[pred_col].values
    preds_in_test = np.array([preds for preds in preds_in_test])

    return ((runs_in_training, preds_in_training),
            (runs_in_validation, preds_in_validation),
            (runs_in_test, preds_in_test))


def get_ensemble_preds_from_cv_results(predictions, outcomes,
                                       aggregate_col=None,
                                       id_col="id",
                                       ensemble_method=np.mean,
                                       output_dir=None):
    """
    Parameters
    ----------
    predictions: dict
        nested dictionary with keys for cv-repetition, cv-fold and value a pd.DataFrame
    outcomes: dict
        mapping patient id to survival outcome
    aggregate_col: str
        name of the column of each prediction dataframe within predictions that holds the
        patient prediction of a single run which will be further used to construct an ensemble
        prediction.
        If None, the column starting with 'pred_per_pat' will be chosen.
    ensemble_method: callable
        a numpy function for computing an ensemble prediction from the single predictions of all
        the runs (e.g. np.mean or np.median).
    output_dir: str
        path to directory to store a dataframe for each patient with all
        the predictions of the different models as csv file.
        If None, no csv will be stored.
    """

    train_preds_ensemble = []
    train_labels = []

    valid_preds_ensemble = []
    valid_labels = []

    test_preds_ensemble = []
    test_labels = []

    patient_dfs = df_for_each_patient(predictions, id_col=id_col)
    for pat in patient_dfs:
        df = patient_dfs[pat]
        if output_dir is not None:
            df.to_csv(os.path.join(output_dir, pat + "_ensemble_predictions.csv"), index=False)

        if aggregate_col is None:
            aggregate_col = [c for c in df.columns.values
                             if c.startswith("pred_per_pat")][0]
        # print("Using column {} for ensemble prediction and aggregate via {}".format(
        #     aggregate_col, ensemble_method))
        # filter all the runs for which this patient was in training/validation/test
        (_, preds_in_training), (_, preds_in_validation), (_, preds_in_test) = predictions_by_cohort_for_single_patient(
            df, pred_col=aggregate_col)

        # the prediction on aggregate level for all those samples
        ensemble_pred_training = ensemble_method([p for p in preds_in_training])
        if not np.isnan(ensemble_pred_training):
            train_preds_ensemble.append(ensemble_pred_training)
            train_labels.append(outcomes[pat])

        ensemble_pred_validation = ensemble_method([p for p in preds_in_validation])
        if not np.isnan(ensemble_pred_validation):
            valid_preds_ensemble.append(ensemble_pred_validation)
            valid_labels.append(outcomes[pat])

        ensemble_pred_test = ensemble_method([p for p in preds_in_test])
        if not np.isnan(ensemble_pred_test):
            test_preds_ensemble.append(ensemble_pred_test)
            test_labels.append(outcomes[pat])

        # print("{} was part of {} training, {} validation and {} test runs.".format(
        #     pat, len(preds_in_training), len(preds_in_validation),
        #     len(preds_in_test)))

    train_preds_ensemble = np.array(train_preds_ensemble)
    train_labels = np.array(train_labels)

    valid_preds_ensemble = np.array(valid_preds_ensemble)
    valid_labels = np.array(valid_labels)

    # this might potentially not contain values, but we still have to
    # reshape it to have the proper number of dimensions
    test_preds_ensemble = np.array(test_preds_ensemble).reshape((-1,) + train_preds_ensemble.shape[1:])
    test_labels = np.array(test_labels).reshape((-1,) + train_labels.shape[1:])

    return ((train_preds_ensemble, train_labels),
            (valid_preds_ensemble, valid_labels),
            (test_preds_ensemble, test_labels))


def evaluate_ensemble_auc(train_preds_ensemble, train_labels,
                          valid_preds_ensemble, valid_labels,
                          test_preds_ensemble, test_labels,
                          output_dir=None, class_thresholds=[0.5]):

    auc_train_ensemble = roc_auc_score(train_labels, train_preds_ensemble)
    auc_valid_ensemble = roc_auc_score(valid_labels, valid_preds_ensemble)

    # print("ENSEMBLE train set: AUC:{}".format(auc_train_ensemble))
    # print("ENSEMBLE valid set: AUC:{}".format(auc_valid_ensemble))

    # for the test set as well
    if len(test_preds_ensemble) > 0:
        auc_test_ensemble = roc_auc_score(test_labels, test_preds_ensemble)
    else:
        test_preds_ensemble = None
        test_labels = None
        auc_test_ensemble = None

    # print("ENSEMBLE test set: AUC: {}".format(auc_test_ensemble))

    if output_dir is not None:
        plot_auc_curves(
            train_preds_ensemble, train_labels,
            valid_preds_ensemble, valid_labels,
            test_preds_ensemble, test_labels,
            subtitle=None,
            save_dir=os.path.join(output_dir, "auc.png"))

        # convert prediction to class label based on the given
        # class thresholds
        plot_confusion_matrices(
            np.digitize(train_preds_ensemble, class_thresholds), train_labels,
            np.digitize(valid_preds_ensemble, class_thresholds), valid_labels,
            np.digitize(test_preds_ensemble, class_thresholds), test_labels,
            subtitle=None,
            save_dir=os.path.join(output_dir, "confusion_matrices_pred_cutoff={}.png".format(
                str(class_thresholds))))

    return pd.DataFrame({
                "Training AUC": [auc_train_ensemble],
                "Validation AUC": [auc_valid_ensemble],
                "Test AUC": [auc_test_ensemble]
            })


def evaluate_ensemble_cindex(train_preds_ensemble, train_labels,
                             valid_preds_ensemble, valid_labels,
                             test_preds_ensemble, test_labels,
                             output_dir=None):

    # use same label for training and validation since the validation set
    # was part of the training set in all the splits
    ci_train_ensemble = compute_ci(train_preds_ensemble, train_labels)
    ci_valid_ensemble = compute_ci(valid_preds_ensemble, valid_labels)

    # print("ENSEMBLE train set: concordance index: {}".format(ci_train_ensemble))
    # print("ENSEMBLE valid set: concordance index: {}".format(ci_valid_ensemble))

    # for the test set as well
    if len(test_preds_ensemble) > 0:
        ci_test_ensemble = compute_ci(test_preds_ensemble, test_labels)
    else:
        test_preds_ensemble = None
        test_labels = None
        ci_test_ensemble = None

    # print("ENSEMBLE test set: concordance index: {}".format(ci_test_ensemble))

    if output_dir is not None:
        # plot ensemble kaplan meier
        _, p_vals = plot_kms(
            pred_train=train_preds_ensemble,
            train_labels=train_labels,
            pred_valid=valid_preds_ensemble,
            valid_labels=valid_labels,
            pred_test=test_preds_ensemble,
            test_labels=test_labels,
            subtitle=None,
            save_dir=os.path.join(output_dir, "ensemble_kaplan_meier.png"))

        # plot ensemble risk vs survival (with c indices)
        plot_surv_times_vs_risk(
            pred_train=train_preds_ensemble,
            train_labels=train_labels,
            pred_valid=valid_preds_ensemble,
            valid_labels=valid_labels,
            pred_test=test_preds_ensemble,
            test_labels=test_labels,
            subtitle=None,
            save_dir=os.path.join(output_dir, "ensemble_risk_vs_survival.png"))

    return pd.DataFrame({
                "Training CI": [ci_train_ensemble],
                "Validation CI": [ci_valid_ensemble],
                "Test CI": [ci_test_ensemble],
                "Training p_val": [p_vals[0]],
                "Validation p_val": [p_vals[1]],
                "Test p_val": [p_vals[2]]
            })
