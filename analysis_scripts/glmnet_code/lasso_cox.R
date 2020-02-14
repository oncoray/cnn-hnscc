library(glmnet)
library(survival)

perform_lasso_cox <- function(df_train, df_valid, df_test, output_dir){
    #the first 5 columns are "id", "slice_idx", "LRCtime", "LRC", "cohort" and do not contain
    # features
    features_train <- as.matrix(df_train[, -c(1:5)])
    lab_train <- Surv(df_train$LRCtime, df_train$LRC)

    features_test <- as.matrix(df_test[, -c(1:5)])
    lab_test <- Surv(df_test$LRCtime, df_test$LRC)

    # fit cox regression with lasso penalty (cross-validated to find best lambda value)
    # on the training data
    cv_fit <- cv.glmnet(
        features_train, lab_train,
        family = "cox", maxit = 5000,
        intercept=FALSE,
        # alpha=1 => only L1 norm, alpha=0 => only 0.5*L2 norm  (all multiplied by lambda)
        alpha=1, # regularization factor between L1 and L2 (alpha * L1_norm + .5*(1-alpha)*L2_norm) which is then multiplied by lambda
        nfolds=10
    )
    # do not use the best lambda but the one apart from that one standard deviation
    pred_train = data.frame(glm_prediction=predict(cv_fit, newx=features_train, s="lambda.1se")[, 1])
    pred_test = data.frame(glm_prediction=predict(cv_fit, newx=features_test, s="lambda.1se")[, 1])

    # organizatorial information + prediction
    pred_df_train = cbind(df_train[, c(1:5)], pred_train)
    pred_df_test = cbind(df_test[, c(1:5)], pred_test)

    # write out prediction files
    data.table::fwrite(pred_df_train, file.path(output_dir, "glm_pred_train.csv"))
    data.table::fwrite(pred_df_test, file.path(output_dir, "glm_pred_test.csv"))

    # do the same for a validation set if we have one
    if(!missing(df_valid))
    {
        features_valid <- as.matrix(df_valid[, -c(1:5)])
        lab_valid <- Surv(df_valid$LRCtime, df_valid$LRC)
        pred_valid = data.frame(glm_prediction=predict(cv_fit, newx=features_valid, s="lambda.1se")[, 1])
        pred_df_valid = cbind(df_valid[, c(1:5)], pred_valid)
        data.table::fwrite(pred_df_valid, file.path(output_dir, "glm_pred_valid.csv"))
    }
}
