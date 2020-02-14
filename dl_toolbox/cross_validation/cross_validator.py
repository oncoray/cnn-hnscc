import numpy as np
import os
import pandas as pd

from keras import backend as K

from .ensemble import get_ensemble_preds_from_cv_results,\
    evaluate_ensemble_cindex
from ..visualization.ensemble import plot_performance_across_trainings,\
    ensemble_boxplots
from ..preprocessing.stratification import repeated_stratified_cv_splits
from ..utils import check_cohort_differences


class CrossValidator(object):
    def __init__(self, cv_context, reps=1, folds=5):
        self.cv_context = cv_context
        self.reps = reps
        self.folds = folds

        self.cv_splits = None

    def create_cv_splits(self, training_ids):
        self.cv_splits = repeated_stratified_cv_splits(
            training_ids,
            events=[self.cv_context.data_handler.outcome_dict[pat][1]
                    for pat in training_ids],
            cv_k=self.folds, cv_reps=self.reps,
            seed=self.cv_context.seed)

    def store_cv_splits(self, output_dir):
        for r in range(len(self.cv_splits)):
            rep = "rep_" + str(r)
            rep_dir = os.path.join(output_dir, rep)
            os.makedirs(rep_dir, exist_ok=False)
            for k in range(len(self.cv_splits[r])):
                fold = "fold_" + str(k)
                fold_dir = os.path.join(rep_dir, fold)
                os.makedirs(fold_dir, exist_ok=False)

                # store the training and validation split here
                ids_t, ids_v = self.cv_splits[r][k]
                pd.DataFrame(ids_t).to_csv(
                    os.path.join(fold_dir, "ids_training.csv"),
                    index=False, header=False)
                pd.DataFrame(ids_v).to_csv(
                    os.path.join(fold_dir, "ids_validation.csv"),
                    index=False, header=False)

    def make_train_test_split(self):
        all_ids = self.cv_context.data_handler.patient_ids
        train_ids = self.cv_context.get_training_ids()
        test_ids = list(set(all_ids) - set(train_ids))

        print("Use {} patients for training/validation and {} for testing".format(
            len(train_ids), len(test_ids)))

        if len(test_ids) == 0:
            test_ids = None

        return train_ids, test_ids

    def run_cross_validation(self, output_dir=None):
        """
        The main function that calls most of the other interface functions.
        """
        # create the output directory for storing stuff
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=False)
        else:
            output_dir = "./cv_output"

        # be sure that the context has its data ready
        self.cv_context.data_handler.read_data_if_necessary()

        # allow the cv context to write additional information
        # about the data
        self.cv_context.write_data_info(output_dir)

        # make training/test split
        train_ids, test_ids = self.make_train_test_split()

        if test_ids is not None:
            # check if training and testset are similar wrt survival information
            assert not set(train_ids).intersection(set(test_ids))
            check_cohort_differences(
                train_ids, test_ids, self.cv_context.data_handler.outcome_dict)

        else:
            print("No separate testset available!")

        print("\nStarting {}-fold cross-validation with {} repetitions.".format(
            self.folds, self.reps))

        # for each of the training repetitions create stratified cross validation splits
        # of the training data
        self.create_cv_splits(train_ids)
        # store the cv splits and create folders for each rep and fold
        self.store_cv_splits(output_dir)

        perf_dfs = []  # dataframes of performance indices for each fold
        pred_dfs = dict()
        for r in range(self.reps):
            rep = "rep_" + str(r)
            rep_dir = os.path.join(output_dir, rep)
            # os.makedirs(rep_dir, exist_ok=False)

            for k in range(self.folds):
                fold = "fold_" + str(k)
                fold_dir = os.path.join(rep_dir, fold)
                # os.makedirs(fold_dir, exist_ok=False)

                ids_train, ids_valid = self.cv_splits[r][k]

                pred_dict, perf_df = self.cv_context.run_single_fold(
                    ids_train, ids_valid, test_ids,
                    output_dir=fold_dir, rep=r, fold=k,
                    print_model_summary=((r==0) and (k==0)))

                perf_dfs.append(perf_df)
                # update the pred_dfs dictionary
                # with the dictionary pred_dict that contains
                # the subevaluation results for the current fold
                for res_descr in pred_dict:
                    if res_descr not in pred_dfs:
                        pred_dfs[res_descr] = {}

                    if r not in pred_dfs[res_descr]:
                        pred_dfs[res_descr][r] = {}

                    pred_dfs[res_descr][r][k] = pred_dict[res_descr]

                # release GPU memory after every training
                K.clear_session()

        # all model performances for different folds and reps
        full_perf_df = pd.concat(perf_dfs, ignore_index=True, sort=False)
        # store the evaluation perf_df
        full_perf_df.to_csv(
            os.path.join(output_dir, "model_performances.csv"),
            na_rep="NAN", index=False)

        # store the test set
        pd.DataFrame(test_ids).to_csv(
            os.path.join(output_dir, "ids_test.csv"), index=False,
            header=False)

        return pred_dfs, perf_dfs

    def _evaluate_ensemble_performance(self, pred_dfs, ensemble_metric_fns,
                                       ensemble_method=np.mean,
                                       output_dir=None):
        # evaluate ensemble and consistency across trainings for all different kinds
        # of prediction that was done
        ensemble_results = {}
        for res_descr, pred_dict in pred_dfs.items():
            print("\nEnsemble evaluation for {}".format(res_descr))
            # pred dict is a nested dictionary with rep/fold and as value a pd.DataFrame
            #
            # create a plot for each patient with predictions over the different models
            out_path = os.path.join(output_dir, "ensemble_" + res_descr)
            os.makedirs(out_path, exist_ok=True)
            # consistency_path = os.path.join(out_path, "consistency")
            # os.makedirs(consistency_path, exist_ok=True)

            # plot_performance_across_trainings(
            #     pred_dict, self.cv_context.data_handler.labels,
            #     save_dir=consistency_path)

            ensemble_boxplots(
                pred_dict, self.cv_context.data_handler.outcome_dict,
                id_col=self.cv_context.data_handler.id_col,
                time_col=self.cv_context.data_handler.time_col,
                event_col=self.cv_context.data_handler.event_col,
                output_dir=out_path)

            ensemble_res = get_ensemble_preds_from_cv_results(
                pred_dict, self.cv_context.data_handler.labels,
                aggregate_col=None,
                id_col=self.cv_context.data_handler.id_col,
                ensemble_method=ensemble_method,
                output_dir=out_path)

            train_preds_ensemble, train_labels = ensemble_res[0]
            valid_preds_ensemble, valid_labels = ensemble_res[1]
            test_preds_ensemble, test_labels = ensemble_res[2]
            # print(train_preds_ensemble.shape, train_labels.shape)
            # print(valid_preds_ensemble.shape, valid_labels.shape)
            # print(test_preds_ensemble.shape, test_labels.shape)
            metrics = []
            for metric_fn in ensemble_metric_fns:
                metric_df = metric_fn(
                    train_preds_ensemble, train_labels,
                    valid_preds_ensemble, valid_labels,
                    test_preds_ensemble, test_labels,
                    output_dir=out_path)

                metrics.append(metric_df)

            all_metrics = pd.concat(metrics, axis=1)
            all_metrics["Ensemble method"] = [ensemble_method.__name__]
            all_metrics.to_csv(
                os.path.join(out_path, "model_performances.csv"), index=False)
            ensemble_results[res_descr] = all_metrics

        return ensemble_results

    def evaluate_ensemble_performance(self, pred_dfs, ensemble_method=np.mean,
                                      ensemble_metric_fns=[
                                          evaluate_ensemble_cindex],
                                      output_dir=None):

        return self._evaluate_ensemble_performance(
            pred_dfs, ensemble_metric_fns=ensemble_metric_fns,
            ensemble_method=ensemble_method,
            output_dir=output_dir)
