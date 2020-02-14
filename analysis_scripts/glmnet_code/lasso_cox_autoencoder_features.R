source("lasso_cox.R")


base_dir = "/home/MED/starkeseb/my_experiments/paper_evaluation_of_dl_approaches/autoencoder/"
output_base = "/home/MED/starkeseb/my_experiments/paper_evaluation_of_dl_approaches/autoencoder/glmnet_performance"

reps = list.files(base_dir, recursive=FALSE, include.dirs=TRUE, pattern="rep_*")

for(rep in reps){
    rep_dir = file.path(base_dir, rep)
    folds = list.files(rep_dir, recursive=FALSE, include.dirs=TRUE, pattern="fold_*")
    for(fold in folds){
        input_path = file.path(rep_dir, fold)
        print(input_path)

        # load data for training, validation and test
        df_train = data.table::fread(file.path(input_path, "encoder_features_training.csv"))
        df_valid = data.table::fread(file.path(input_path, "encoder_features_validation.csv"))
        df_test = data.table::fread(file.path(input_path, "encoder_features_test.csv"))

        output_path = file.path(output_base, rep, fold)
        # create output_path if it does not exist
        if(!dir.exists(output_path)){dir.create(output_path, recursive=TRUE)}

        # do lasso-cox on the training set and write out predictions
        perform_lasso_cox(df_train, df_valid, df_test, output_path)
    }
}


