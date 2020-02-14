import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from ..cross_validation.ensemble import df_for_each_patient,\
    predictions_by_cohort_for_single_patient


def ensemble_boxplots(pred_dict, outcome_dict, id_col="id", time_col="LRCtime",
                      event_col="LRC", pred_col=None,
                      cohort_col="cohort", output_dir=None,
                      show_ids=False):
    """
    Generates boxplots for patients indicating the distribution
    of predictions across ensemble models
    (e.g. models from different cross-validation splits).
    Boxplots of patients will be aligned by increasing event-time
    and two plots will be given, for patients with and without observed
    event.

    Parameters
    ----------
    pred_dict: nested dict of pandas.DataFrame
        Keys are for each repetition and each cv fold and the value is a pd.DataFrame.
        That df contains at least the columns <id_col>, 'rep', 'fold', <cohort_col>, <pred_col>
        which contain predictions for each patient during each fold of the cross-validation
        runs over which the ensemble will be computed over and the cohort that each patient
        was assigned to during each of the model trainings.
    outcome_dict: dict
        contains the event-time and event status for each patient id.
        Is converted to a pd.DataFrame with columns
        <id_col>, <time_col> and <event_col> for each patient.
    """
    if pred_col is None:
        r0 = list(pred_dict.keys())[0]
        f0 = list(pred_dict[r0].keys())[0]
        pred_col = [c for c in pred_dict[r0][f0].columns if c.startswith("pred_per_pat")][0]
    print(f"ensemble_boxplots: use column {pred_col} for predictions!")

    # create one large dataframe with all predictions of all folds
    dfs = []
    for r in pred_dict:
        for f in pred_dict[r]:
            df = pred_dict[r][f]
            df = df[[id_col, "rep", "fold", cohort_col, pred_col]]
            dfs.append(df)
            # print(r, f, df.shape)
    dfs = pd.concat(dfs, axis=0)

    # create the outcome as dataframe
    ids = sorted(list(outcome_dict.keys()))
    outcome_df = pd.DataFrame({
        id_col: ids,
        time_col: [outcome_dict[id][0] for id in ids],
        event_col: [outcome_dict[id][1] for id in ids]})

    dfs = dfs.merge(outcome_df, on=id_col)
    # print("dfs=\n", dfs)

    figs = []
    for cohort in pd.unique(dfs[cohort_col]):
        # draw a plot that shows for all the patients of the cohort their predictions
        preds = {}
        df = dfs[dfs[cohort_col] == cohort]
        # print(cohort, f"df.shape={df.shape}")
        for _, row in df.iterrows():
            id = row[id_col]
            pred = row[pred_col]

            if id not in preds:
                preds[id] = [pred]
            else:
                preds[id].append(pred)

        for pat in preds:
            preds[pat] = np.array(preds[pat])
            # print(cohort, pat, preds[pat].shape)

        # sort by event status and increasing event time
        width = 12
        height = 7
        f, ax = plt.subplots(2, 1, figsize=(width, height), dpi=300)
        # plots for different event status
        for i in range(2):
            # sort by increasing time
            sub_df = df[df[event_col] == i]
            # print(cohort, f"i={i}, sub_df.shape={sub_df.shape}")
            sub_df = sub_df.sort_values(by=time_col)
            # print(i, df)
            # this respects order!
            local_ids = pd.unique(sub_df[id_col])
            # print("len(local_ids)", len(local_ids))
            vals = np.array(
                [sub_df.loc[sub_df[id_col] == id, [pred_col]].values.squeeze()
                 for id in local_ids])
            # print("vals.shape", vals.shape)
            if vals.ndim == 1:
                vals = np.expand_dims(vals, -1)
            # print(f"local_ids.shape={local_ids.shape}, vals.shape={vals.shape}",
            #       f"times.shape={times.shape}")

            ax[i].boxplot([r for r in vals], sym="", showmeans=True)
            # print([local_ids[k] for k in range(len(local_ids))])
            # print(["t={:.2f}".format(times[k]) for k in range(len(local_ids))])
            if show_ids:
                xlabels = [id + ": t={:.2f}".format(outcome_dict[id][0])
                           for id in local_ids]
            else:
                xlabels = ["{:.2f}".format(outcome_dict[id][0])
                           for id in local_ids]
            ax[i].set_xticklabels(np.array(xlabels), rotation=90)
            ax[i].set_xlabel("Patient times")
            # ax[i].set_ylim(-1.5, 1.5)
            ax[i].set_ylabel("Prediction")
            title = f"{cohort.capitalize()} cohort: ensemble for " + ("censored patients" if i==0 else "patients with event")
            ax[i].set_title(title)

        plt.tight_layout()
        if output_dir is not None:
            out_file = os.path.join(output_dir, f"boxplot_{cohort}_predictions.png")
            f.savefig(out_file)
            print(f"\nstored boxplot to {out_file}.")

        figs.append(f)

    if output_dir is None:
        plt.show()

    return figs


def boxplots(ax, pat, pat_preds, pat_label, cohort, xlabels, rotation=None):
        """
        Parameters
        ----------
        pat_preds: np.array(2D) of format n_model_runs x n_slices
        """
        # pat_preds should be n_models x n_slices

        # print(len(xlabels), pat_preds.shape)
        # boxplot for each column -> have to transpose
        ax.boxplot(pat_preds.T, notch=False, showmeans=True)
        #df = pd.DataFrame({
        #    'run_id': [xlabels[run_id] for run_id in range(pat_preds.shape[0]) for slice_id in range(pat_preds.shape[1])],
        #    'pred': [pat_preds[run_id, slice_id] for run_id in range(pat_preds.shape[0]) for slice_id in range(pat_preds.shape[1])]
        #})

        #sns.swarmplot(x="run_id", y="pred", data=df, color="white", edgecolor="gray", ax=ax)
        #sns.violinplot(x="run_id", y="pred", data=df, inner=None, ax=ax)

        ax.set_title("{} performance {}\n(label = {})".format(
            cohort.capitalize(), pat, pat_label))
        ax.set_ylabel("predicted value")
        ax.set_ylim([-1.1, 1.1])
        # ax.legend(loc="upper right")

        ax.set_xticks(range(1, len(pat_preds)+1))
        ax.set_xticklabels([])
        if rotation is None:
            rotation = 45 if pat_preds.shape[0] > 10 else 0

        for i, xpos in enumerate(ax.get_xticks()):
            ax.text(xpos, -1.05, xlabels[i], size=12, ha='center', rotation=rotation)


def plot_performance_across_trainings(predictions, outcomes, save_dir=None):
    """
    For each patient, creates boxplots indicating the distribution of sample
    predictions of that patient across ensemble models
    (e.g. models from different cross-validation splits).
    This allows to inspect how sample predictions differ for each patient
    within a single model and across models.
    Also, for each patient, plots are separated by the cohort the patient
    belonged to (training/validation/test).

    Parameters
    ----------
    predictions: dict
        nested dict with key levels are repetition/label then fold
        and value a pd.DataFrame
    outcomes: dict
        mapping patient ids to an iterable of length two (event_time, event_status)
    """
    patient_dfs = df_for_each_patient(predictions)
    for pat in patient_dfs:
        df = patient_dfs[pat]
        outcome_label = outcomes[pat]
        train, valid, test = predictions_by_cohort_for_single_patient(df)

        labels_train, preds_train = train
        labels_valid, preds_valid = valid
        labels_test, preds_test = test

        #print("{} was part of {} trainings, {} validations and {} tests".format(
        #      pat, len(preds_train), len(preds_valid), len(preds_test)))
        #print(preds_train.shape, preds_valid.shape, preds_test.shape)

        if len(preds_test) == 0:
            # this patient was part of training and validation
            # and we need to create 2 plots (training -> slice/pat, validation-> slice/pat) if everything
            # went well. But it might be that some evaluations failed in some splits and we have
            # less data for a patient (or even no training/no validation samples)
            n=2
            if preds_train.shape[0] <= 0:
                print("[W]: {} has not participated in training splits!".format(pat))
                n -= 1
            if preds_valid.shape[0] <= 0:
                print("[W]: {} has not participated in validation splits!".format(pat))
                n -= 1
            if n <= 0:
                print("[W]: {}: not able to plot data!".format(pat))
                continue

            width = 5 * max(preds_train.shape[0], preds_valid.shape[0])
            height = 12
            f, ax = plt.subplots(2, 1, figsize=(width, height), dpi=300)  # (width, height)

            if preds_train.shape[0] > 0:
                boxplots(ax[0], pat, preds_train,
                         outcome_label, "Training", labels_train, rotation=None)
            if preds_valid.shape[0] > 0:
                boxplots(ax[1], pat, preds_valid,
                         outcome_label, "Validation", labels_valid)
        else:
            # this patient was part of the test set only
            # and we need to create 1 plot (test -> slice/pat)
            f, ax = plt.subplots(1, 1, figsize=(5 * len(preds_test), 6), dpi=300)

            boxplots(ax, pat, preds_test,
                     outcome_label, "Test", labels_test, rotation=None)

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, pat + ".png"))

        plt.close()
