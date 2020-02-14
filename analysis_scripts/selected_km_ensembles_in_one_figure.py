import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dl_toolbox.cross_validation.ensemble import get_ensemble_preds_from_cv_results
from dl_toolbox.data.read import read_outcome
from dl_toolbox.visualization.kaplan_meier import plot_kms

from create_ensemble_manually import read_fold_predictions


result_dir = ("/home/MED/starkeseb/g40fs4-hprt/HPRT-Data/ONGOING_PROJECTS/"
             "Radiomics/RadiomicsAnalysis/Project/DeepRadiomics_Sebastian/"
             "Experiments/paper_evaluation_of_dl_approaches")

# for the scientific reports paper
# model_dirs = [
#     "transfer_learning_densenet201_last_preproc",
#     "from_scratch_cox_tanh_no_bn",
#     "from_scratch_3d_16_sample_per_patient_no_augmentation"
# ]
# # for extension of the ylabel
# labels = [
#     "Transfer learning",
#     "2D-CNN",
#     "3D-CNN"
# ]

# for the MIDL 2020 submission: 2D,
model_dirs = [
    # "from_scratch_2d_64x64_1_sample_per_patient_no_augmentation",
    "from_scratch_2d_64x64_1_sample_per_patient",
    # "from_scratch_2d_64x64_32_sample_per_patient_no_augmentation",
    "from_scratch_2d_64x64_32_sample_per_patient",
]
labels = [
    # "2D-CNN, 1 sample, no augmentation",
    "2D-CNN, 1 sample",
    # "2D-CNN, 32 samples, no augmentation",
    "2D-CNN, 32 samples"
]

# for the MIDL 2020 submission: 3D,
# model_dirs = [
#     "from_scratch_3d_1_sample_per_patient_no_augmentation",
#     # "from_scratch_3d_1_sample_per_patient",
#     "from_scratch_3d_16_sample_per_patient_no_augmentation",
#     # "from_scratch_3d",
# ]
# labels = [
#     "3D-CNN, 1 sample, no augmentation",
#     # "3D-CNN, 1 sample",
#     "3D-CNN, 16 samples, no augmentation",
#     # "3D-CNN, 16 samples"
# ]


output_dir = "/home/MED/starkeseb/tmp/"

outcome_dict = read_outcome(
    "/home/MED/starkeseb/mbro_local/data/DKTK/outcome.csv",
    id_col="ID_Radiomics",
    time_col="LRCtime",
    event_col="LRC")

n_rows = len(model_dirs)
n_cols = 3

width = n_cols * 4
height = n_rows * (5 * .8)

f, axs = plt.subplots(
    n_rows, n_cols, figsize=(width, height), dpi=300, sharey=True)
axs = np.reshape(axs, (n_rows, n_cols))

f2, axs2 = plt.subplots(
    n_rows, n_cols-1, figsize=(width-4, height), dpi=300, sharey=True)
axs2 = np.reshape(axs2, (n_rows, n_cols-1))


for i, model_dir in enumerate(model_dirs):
    path = os.path.join(result_dir, model_dir)

    # for each cross-validation fold a pd.DataFrame
    pred_dict = read_fold_predictions(path)

    ensemble_res = get_ensemble_preds_from_cv_results(
        pred_dict, outcome_dict, aggregate_col=None,
        ensemble_method=np.mean,
        output_dir=None)

    train_preds_ensemble, train_labels = ensemble_res[0]
    valid_preds_ensemble, valid_labels = ensemble_res[1]
    test_preds_ensemble, test_labels = ensemble_res[2]

    # this figure now contains three plots: for training, validation, test
    _, p_vals = plot_kms(
        pred_train=train_preds_ensemble,
        train_labels=train_labels,
        pred_valid=valid_preds_ensemble,
        valid_labels=valid_labels,
        pred_test=test_preds_ensemble,
        test_labels=test_labels,
        time_in_month=True,
        save_dir=None,
        axs=axs[i,:],
        y_label=labels[i] + "\n\nLoco-regional tumour control",
        set_titles=i==0,
        table_below_scaling=0.17)

    # leave out the validation plots and create new figures
    _, p_vals = plot_kms(
        pred_train=train_preds_ensemble,
        train_labels=train_labels,
        pred_valid=test_preds_ensemble,
        valid_labels=test_labels,
        # pred_test=test_preds_ensemble,
        # test_labels=test_labels,
        time_in_month=True,
        save_dir=None,
        axs=axs2[i,:],
        y_label=labels[i] + "\n\nLoco-regional tumour control",
        set_titles=i==0,
        table_below_scaling=0.17)

    axs2[i, 0].set_title("Training")
    axs2[i, 1].set_title("Test")

f.tight_layout()
output_file = os.path.join(output_dir, "ensemble_kaplan_meiers.png")
f.savefig(output_file)
print("KM plot saved to", output_file)

f2.tight_layout()
output_file = os.path.join(output_dir, "ensemble_kaplan_meiers_without_validation.png")
f2.savefig(output_file)
print("KM plot saved to", output_file)
