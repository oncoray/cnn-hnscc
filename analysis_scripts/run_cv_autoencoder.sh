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

python cv_autoencoder.py \
    --input $DKTK_BER $DKTK_DD $DKTK_EU $DKTK_FFM $DKTK_MUC $DKTK_TU $FDG $FMISO $FMISO_TUE $STR_UKD \
    --outcome $BASE_DIR/outcome.csv \
    --id_col ID_Radiomics \
    --time_col LRCtime \
    --event_col LRC \
    --train_id_file $HOME/dktk_train_ids.csv \
    --output $HOME/dl_analysis/autoencoder \
    --batch 32 \
    --epochs 100 \
    --kfold 10 \
    --lrelu 0.01 \
    --batchnorm "" \
    --reps 3 \
    --opti adam \
    --lr 1e-3 \
    --loss binary_crossentropy \
    --nslices 7 8 \
    --time_perturb 0 \
    --shear_range 0.1 \
    --zoom_range 0.1 \
    --rotation_range 45 \
    --fill_mode nearest \
    --pca_dims 1 2 5 10
