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

python cv_transfer_learning.py \
    --input $DKTK_BER $DKTK_DD $DKTK_EU $DKTK_FFM $DKTK_MUC $DKTK_TU $FDG $FMISO $FMISO_TUE $STR_UKD \
    --outcome $BASE_DIR/outcome.csv \
    --id_col ID_Radiomics \
    --time_col LRCtime \
    --event_col LRC \
    --train_id_file $HOME/dktk_train_ids.csv \
    --output $HOME/dl_analysis/transfer_learning_resnet50_activation_37 \
    --architecture ResNet50 \
    --layer activation_37 \
    --batch 32 \
    --epochs 1 20 \
    --kfold 10 \
    --reps 3 \
    --opti adam \
    --lr 5e-4 1e-6 \
    --loss cox \
    --finalact tanh \
    --batchnorm "" \
    --lrelu 0. \
    --l1 0. \
    --l2 0. \
    --nslices 7 8 \
    --no_data_augmentation
