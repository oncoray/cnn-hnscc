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

python cv_train_from_scratch_2d_batchgenerators.py \
    --input $DKTK_BER $DKTK_DD $DKTK_EU $DKTK_FFM $DKTK_MUC $DKTK_TU $FDG $FMISO $FMISO_TUE $STR_UKD \
    --outcome $BASE_DIR/outcome.csv \
    --id_col ID_Radiomics \
    --time_col LRCtime \
    --event_col LRC \
    --train_id_file $HOME/dktk_train_ids.csv \
    --output $HOME/dl_analysis/from_scratch_2d_batchgenerators_augmentation \
    --crop_size 16 224 224 \
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
    --time_perturb 0 \
    --do_rotation \
    --p_rot_per_sample 0.5 \
    --angle_x 15 \
    --angle_y 15 \
    --angle_z 15 \
    --do_elastic_deform \
    --p_el_per_sample 0.5 \
    --deformation_scale 0 0.25 \
    --do_scale \
    --p_scale_per_sample 0.5 \
    --scale 0.75 1.25 \
    --do_mirror \
    --p_per_sample 0.15 \
    --brightness_range 0.7 1.5 \
    --gaussian_noise_variance 0 0.05 \
    --gamma_range 0.5 2 \

