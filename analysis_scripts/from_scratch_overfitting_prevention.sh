
BASE_DIR=$HOME/mbro_local/data/DKTK
DKTK_BER=$BASE_DIR/DKTK_BER/numpy_preprocessed
DKTK_DD=$BASE_DIR/DKTK_DD/numpy_preprocessed
DKTK_EU=$BASE_DIR/DKTK_EU/numpy_preprocessed
DKTK_FFM=$BASE_DIR/DKTK_FFM/numpy_preprocessed
DKTK_MUC=$BASE_DIR/DKTK_MUC/numpy_preprocessed
DKTK_TU=$BASE_DIR/DKTK_TU/numpy_preprocessed
FDG=$BASE_DIR/FDG/numpy_preprocessed
FMISO=$BASE_DIR/FMISO/numpy_preprocessed
FMISO_TUE=$BASE_DIR/FMISO_TUE/numpy_preprocessed
STR_UKD=$BASE_DIR/STR_UKD/numpy_preprocessed

# time perturbation
python cv_train_from_scratch.py \
    --input $DKTK_BER $DKTK_DD $DKTK_EU $DKTK_FFM $DKTK_MUC $DKTK_TU $FDG $FMISO $FMISO_TUE $STR_UKD \
    --outcome $BASE_DIR/outcome.csv \
    --id_col ID_Radiomics \
    --time_col LRCtime \
    --event_col LRC \
    --train_id_file $HOME/dktk_train_ids.csv \
    --output $HOME/dl_analysis/from_scratch_2d_regularization_time_perturb \
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
    --no_data_augmentation


# additional L1 and L2 to baseline dropout
python cv_train_from_scratch.py \
    --input $DKTK_BER $DKTK_DD $DKTK_EU $DKTK_FFM $DKTK_MUC $DKTK_TU $FDG $FMISO $FMISO_TUE $STR_UKD \
    --outcome $BASE_DIR/outcome.csv \
    --id_col ID_Radiomics \
    --time_col LRCtime \
    --event_col LRC \
    --train_id_file $HOME/dktk_train_ids.csv \
    --output $HOME/dl_analysis/from_scratch_2d_regularization_l1_l2 \
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
    --time_perturb 0 \
    --no_data_augmentation


# increase dropout to 0.5
python cv_train_from_scratch.py \
    --input $DKTK_BER $DKTK_DD $DKTK_EU $DKTK_FFM $DKTK_MUC $DKTK_TU $FDG $FMISO $FMISO_TUE $STR_UKD \
    --outcome $BASE_DIR/outcome.csv \
    --id_col ID_Radiomics \
    --time_col LRCtime \
    --event_col LRC \
    --train_id_file $HOME/dktk_train_ids.csv \
    --output $HOME/dl_analysis/from_scratch_2d_regularization_dropout_0.5 \
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
    --time_perturb 0 \
    --no_data_augmentation

# only add data augmentation to baseline dropout
python cv_train_from_scratch.py \
    --input $DKTK_BER $DKTK_DD $DKTK_EU $DKTK_FFM $DKTK_MUC $DKTK_TU $FDG $FMISO $FMISO_TUE $STR_UKD \
    --outcome $BASE_DIR/outcome.csv \
    --id_col ID_Radiomics \
    --time_col LRCtime \
    --event_col LRC \
    --train_id_file $HOME/dktk_train_ids.csv \
    --output $HOME/dl_analysis/from_scratch_2d_regularization_augmentation \
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
    --time_perturb 0 \
    --shear_range 0.2 \
    --zoom_range 0.2 \
    --rotation_range 30 \
    --fill_mode nearest \

# only data augmentation, no dropout (just like we had in estro model)
python cv_train_from_scratch.py \
    --input $DKTK_BER $DKTK_DD $DKTK_EU $DKTK_FFM $DKTK_MUC $DKTK_TU $FDG $FMISO $FMISO_TUE $STR_UKD \
    --outcome $BASE_DIR/outcome.csv \
    --id_col ID_Radiomics \
    --time_col LRCtime \
    --event_col LRC \
    --train_id_file $HOME/dktk_train_ids.csv \
    --output $HOME/dl_analysis/from_scratch_2d_regularization_augmentation_no_dropout \
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
    --time_perturb 0 \
    --shear_range 0.2 \
    --zoom_range 0.2 \
    --rotation_range 30 \
    --fill_mode nearest \


# higher dropout, data augmentation
python cv_train_from_scratch.py \
    --input $DKTK_BER $DKTK_DD $DKTK_EU $DKTK_FFM $DKTK_MUC $DKTK_TU $FDG $FMISO $FMISO_TUE $STR_UKD \
    --outcome $BASE_DIR/outcome.csv \
    --id_col ID_Radiomics \
    --time_col LRCtime \
    --event_col LRC \
    --train_id_file $HOME/dktk_train_ids.csv \
    --output $HOME/dl_analysis/from_scratch_2d_regularization_augmentation_dropout_0.5 \
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
    --time_perturb 0 \
    --shear_range 0.2 \
    --zoom_range 0.2 \
    --rotation_range 30 \
    --fill_mode nearest \


# add higher dropout, data augmentation and l1+l2 regularization
python cv_train_from_scratch.py \
    --input $DKTK_BER $DKTK_DD $DKTK_EU $DKTK_FFM $DKTK_MUC $DKTK_TU $FDG $FMISO $FMISO_TUE $STR_UKD \
    --outcome $BASE_DIR/outcome.csv \
    --id_col ID_Radiomics \
    --time_col LRCtime \
    --event_col LRC \
    --train_id_file $HOME/dktk_train_ids.csv \
    --output $HOME/dl_analysis/from_scratch_2d_regularization_augmentation_l1_l2_dropout_0.5 \
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
    --time_perturb 0 \
    --shear_range 0.2 \
    --zoom_range 0.2 \
    --rotation_range 30 \
    --fill_mode nearest \
