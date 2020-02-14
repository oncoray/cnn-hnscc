
for batch_size in 256 128 64 32
do
    for slices in 23 15 7
    do
        # echo batch_size=$batch_size, slices=${slices}_$((slices+1))
        python cv_train_from_scratch.py /home/MED/starkeseb/mbro_local/data/ DKTK \
            --output /home/MED/starkeseb/g40fs4-hprt/HPRT-Data/ONGOING_PROJECTS/Radiomics/RadiomicsAnalysis/Project/DeepRadiomics_Sebastian/Experiments/paper_evaluation_of_dl_approaches/from_scratch_cox_tanh_no_bn_batch_${batch_size}_slices_${slices}_$((slices+1)) \
            --baseline_feats baseline_features.csv \
            --batch $batch_size \
            --epochs 50 \
            --kfold 10 \
            --reps 3 \
            --opti adam \
            --lr 5e-5 \
            --loss cox \
            --finalact tanh \
            --batchnorm "" \
            --lrelu 0. \
            --nslices $slices $((slices+1)) \
            --time_perturb 0 \
            --shear_range 0 \
            --zoom_range 0 \
            --rotation_range 0 \
            --fill_mode nearest \
            --train_id_file /home/MED/starkeseb/dktk_train_ids.csv
    done
done
