
# time perturbation
python cv_train_from_scratch.py /home/MED/starkeseb/mbro_local/data/ DKTK \
    --output /home/MED/starkeseb/g40fs4-hprt/HPRT-Data/ONGOING_PROJECTS/Radiomics/RadiomicsAnalysis/Project/DeepRadiomics_Sebastian/Experiments/paper_evaluation_of_dl_approaches/from_scratch_time_perturbation \
    --batch 32 \
    --epochs 50 \
    --kfold 10 \
    --reps 3 \
    --opti adam \
    --lr 5e-5 \
    --loss cox \
    --finalact tanh \
    --batchnorm "" \
    --lrelu 0. \
    --dropout 0.3 \
    --l1 0. \
    --l2 0. \
    --nslices 7 8 \
    --time_perturb 0.1 \
    --shear_range 0 \
    --zoom_range 0 \
    --rotation_range 0 \
    --fill_mode nearest \
    --train_id_file /home/MED/starkeseb/dktk_train_ids.csv


# additional L1 and L2 to baseline dropout
python cv_train_from_scratch.py /home/MED/starkeseb/mbro_local/data/ DKTK \
    --output /home/MED/starkeseb/g40fs4-hprt/HPRT-Data/ONGOING_PROJECTS/Radiomics/RadiomicsAnalysis/Project/DeepRadiomics_Sebastian/Experiments/paper_evaluation_of_dl_approaches/from_scratch_l1_l2_1e-5 \
    --batch 32 \
    --epochs 50 \
    --kfold 10 \
    --reps 3 \
    --opti adam \
    --lr 5e-5 \
    --loss cox \
    --finalact tanh \
    --batchnorm "" \
    --lrelu 0. \
    --dropout 0.3 \
    --l1 1.e-5 \
    --l2 1.e-5 \
    --nslices 7 8 \
    --time_perturb 0. \
    --shear_range 0 \
    --zoom_range 0 \
    --rotation_range 0 \
    --fill_mode nearest \
    --train_id_file /home/MED/starkeseb/dktk_train_ids.csv

# increase dropout to 0.5
python cv_train_from_scratch.py /home/MED/starkeseb/mbro_local/data/ DKTK \
    --output /home/MED/starkeseb/g40fs4-hprt/HPRT-Data/ONGOING_PROJECTS/Radiomics/RadiomicsAnalysis/Project/DeepRadiomics_Sebastian/Experiments/paper_evaluation_of_dl_approaches/from_scratch_dropout_0.5 \
    --batch 32 \
    --epochs 50 \
    --kfold 10 \
    --reps 3 \
    --opti adam \
    --lr 5e-5 \
    --loss cox \
    --finalact tanh \
    --batchnorm "" \
    --lrelu 0. \
    --dropout 0.5 \
    --l1 0. \
    --l2 0. \
    --nslices 7 8 \
    --time_perturb 0. \
    --shear_range 0 \
    --zoom_range 0 \
    --rotation_range 0 \
    --fill_mode nearest \
    --train_id_file /home/MED/starkeseb/dktk_train_ids.csv

# only add data augmentation to baseline dropout
python cv_train_from_scratch.py /home/MED/starkeseb/mbro_local/data/ DKTK \
    --output /home/MED/starkeseb/g40fs4-hprt/HPRT-Data/ONGOING_PROJECTS/Radiomics/RadiomicsAnalysis/Project/DeepRadiomics_Sebastian/Experiments/paper_evaluation_of_dl_approaches/from_scratch_data_augmentation \
    --batch 32 \
    --epochs 50 \
    --kfold 10 \
    --reps 3 \
    --opti adam \
    --lr 5e-5 \
    --loss cox \
    --finalact tanh \
    --batchnorm "" \
    --lrelu 0. \
    --dropout 0.3 \
    --l1 0. \
    --l2 0. \
    --nslices 7 8 \
    --time_perturb 0. \
    --shear_range 0.2 \
    --zoom_range 0.2 \
    --rotation_range 30 \
    --fill_mode nearest \
    --train_id_file /home/MED/starkeseb/dktk_train_ids.csv

# only data augmentation, no dropout (just like we had in estro model)
python cv_train_from_scratch.py /home/MED/starkeseb/mbro_local/data/ DKTK \
    --output /home/MED/starkeseb/g40fs4-hprt/HPRT-Data/ONGOING_PROJECTS/Radiomics/RadiomicsAnalysis/Project/DeepRadiomics_Sebastian/Experiments/paper_evaluation_of_dl_approaches/from_scratch_data_augmentation_no_dropout \
    --batch 32 \
    --epochs 50 \
    --kfold 10 \
    --reps 3 \
    --opti adam \
    --lr 5e-5 \
    --loss cox \
    --finalact tanh \
    --batchnorm "" \
    --lrelu 0. \
    --dropout 0. \
    --l1 0. \
    --l2 0. \
    --nslices 7 8 \
    --time_perturb 0. \
    --shear_range 0.2 \
    --zoom_range 0.2 \
    --rotation_range 30 \
    --fill_mode nearest \
    --train_id_file /home/MED/starkeseb/dktk_train_ids.csv


# higher dropout, data augmentation
python cv_train_from_scratch.py /home/MED/starkeseb/mbro_local/data/ DKTK \
    --output /home/MED/starkeseb/g40fs4-hprt/HPRT-Data/ONGOING_PROJECTS/Radiomics/RadiomicsAnalysis/Project/DeepRadiomics_Sebastian/Experiments/paper_evaluation_of_dl_approaches/from_scratch_data_augmentation_dropout_0.5 \
    --batch 32 \
    --epochs 50 \
    --kfold 10 \
    --reps 3 \
    --opti adam \
    --lr 5e-5 \
    --loss cox \
    --finalact tanh \
    --batchnorm "" \
    --lrelu 0. \
    --dropout 0.5 \
    --l1 0. \
    --l2 0. \
    --nslices 7 8 \
    --time_perturb 0. \
    --shear_range 0.2 \
    --zoom_range 0.2 \
    --rotation_range 30 \
    --fill_mode nearest \
    --train_id_file /home/MED/starkeseb/dktk_train_ids.csv


# add higher dropout, data augmentation and l1+l2 regularization
python cv_train_from_scratch.py /home/MED/starkeseb/mbro_local/data/ DKTK \
    --output /home/MED/starkeseb/g40fs4-hprt/HPRT-Data/ONGOING_PROJECTS/Radiomics/RadiomicsAnalysis/Project/DeepRadiomics_Sebastian/Experiments/paper_evaluation_of_dl_approaches/from_scratch_data_augmentation_dropout_0.5_l1_l2_1e-5 \
    --batch 32 \
    --epochs 50 \
    --kfold 10 \
    --reps 3 \
    --opti adam \
    --lr 5e-5 \
    --loss cox \
    --finalact tanh \
    --batchnorm "" \
    --lrelu 0. \
    --dropout 0.5 \
    --l1 1.e-5 \
    --l2 1.e-5 \
    --nslices 7 8 \
    --time_perturb 0. \
    --shear_range 0.2 \
    --zoom_range 0.2 \
    --rotation_range 30 \
    --fill_mode nearest \
    --train_id_file /home/MED/starkeseb/dktk_train_ids.csv
