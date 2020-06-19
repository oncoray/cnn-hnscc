import os
import pandas as pd
import numpy as np

def create_prediction_filelist(experiment_dir, ids):
    ensemble_predictions_path = os.path.join(experiment_dir, "ensemble_predictions")
    filelist = [os.path.join(ensemble_predictions_path, f)
                    for f in os.listdir(ensemble_predictions_path)
                    if f.endswith("_ensemble_predictions.csv")
                    and f.split("_ensemble_predictions")[0] in ids
                ]
    return filelist


def combine_prediction_files(prediction_files, outcome_df, pred_col, exp_name):

    ensemble_preds = [None] * len(prediction_files)

    for i, f in enumerate(prediction_files):
        # one file for each patient with results over all 30 runs
        df = pd.read_csv(f)
        df = df.set_index("id")
        pat_id = df.index.values[0]

        df = df[["cohort", pred_col]]
        # for test patients we get one row since they only appear in the test set
        # but for exploratory patients we get two rows since they appear in training + validation
        df = df.groupby(["id", "cohort"]).mean()

        print("\n", pat_id, "df=\n", df)

        ensemble_preds[i] = df

    pred_df = pd.concat(ensemble_preds)
    print("\n", pred_df)
    # also join the outcome
    pred_df = pred_df.join(outcome_df, on="id")

    return pred_df

if __name__ == "__main__":

    id_col = "ID_Radiomics"
    time_col = "LRCtime"
    event_col = "LRC"
    output_base = "/home/MED/starkeseb/tmp"

    outcome = pd.read_csv("/home/MED/starkeseb/mbro_local/data/DKTK/outcome.csv", sep=";")
    outcome = outcome[[id_col, time_col, event_col]]
    outcome = outcome.set_index(id_col)
    print(outcome)

    base_dir = "/home/MED/starkeseb/my_experiments/paper_evaluation_of_dl_approaches/"

    ids = pd.read_csv("/home/MED/starkeseb/dktk_ids.csv", header=None).values.squeeze()
    print("ids has length", len(ids))
    print(ids)

    exp_dirs = [
        # 3D
        "from_scratch_3d_16_sample_per_patient_no_augmentation",
        # 2D
        "from_scratch_cox_linear_bn",
        "from_scratch_cox_linear_no_bn",
        "from_scratch_cox_tanh_bn",
        "from_scratch_cox_tanh_no_bn",
        "from_scratch_with_clinical_volume_cox_tanh_bn",
        # transfer_learning
        "transfer_learning_densenet201_last_preproc",
        "transfer_learning_densenet201_conv4_block48_concat_preproc",
        "transfer_learning_inception_resnet_v2_last_preproc",
        "transfer_learning_inception_resnet_v2_block17_10_ac_preproc",
        "transfer_learning_resnet50_last_preproc",
        "transfer_learning_resnet50_activation_37_preproc",
        ]
    dfs = [None] * len(exp_dirs)

    for i, exp_dir in enumerate(exp_dirs):
        # 1) collect the prediction files for each patient of the test cohort
        filelist_patients = create_prediction_filelist(
            os.path.join(base_dir, exp_dir), ids=ids)

        print("filelist has length", len(filelist_patients))
        # 2) combine the predictions for each patient into a single dataframe
        ensemble_pred_df = combine_prediction_files(
            filelist_patients, outcome_df=outcome, pred_col="pred_per_pat(mean)",
            exp_name=exp_dir)

        exp_outdir = os.path.join(output_base, "ensemble_" + exp_dir)
        os.makedirs(exp_outdir, exist_ok=True)

        # now split into training, validation and test
        ensemble_train = ensemble_pred_df.xs("training", level="cohort")
        ensemble_valid = ensemble_pred_df.xs("validation", level="cohort")
        ensemble_test = ensemble_pred_df.xs("test", level="cohort")

        ensemble_train.to_csv(os.path.join(exp_outdir, f"ensemble_train.csv"))
        ensemble_valid.to_csv(os.path.join(exp_outdir, f"ensemble_valid.csv"))
        ensemble_test.to_csv(os.path.join(exp_outdir, f"ensemble_test.csv"))

        ensemble_pred_df[""]
        dfs[i] = ensemble_pred_df


