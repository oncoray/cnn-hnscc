import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from keras import Model, layers, callbacks, optimizers
from keras import backend as K

from ..result_analysis.evaluate_model import plot_km_and_scatter, compute_cis,\
    compute_pvals
from ..result_analysis.prediction_per_patient import model_predictions_per_patient

from ..losses import neg_cox_log_likelihood
from ..visualization.training_curve import plot_histories
from ..baseline_estimators import CoxModel, RegressionModel


class SurvivalCVContext(object):
    def __init__(self,
                 data_handler,
                 seed=1,
                 train_ids=None,
                 train_fraction=None,
                 epochs=[10],
                 optimizer_cls=optimizers.Adam,
                 lr=[1.e-3],
                 loss=neg_cox_log_likelihood,
                 **kwargs):

        self.data_handler = data_handler
        # set a random seed for reproducible results
        self.seed = seed
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        print("np.random.seed and tf.random.seed set to", self.seed)

        self.train_ids = train_ids
        self.train_fraction = train_fraction

        if self.train_ids is not None and self.train_fraction is not None:
            raise ValueError(
                "Only one of train_ids or train_fraction can be set!")
        elif self.train_ids is None and self.train_fraction is None:
            raise ValueError(
                "train_ids and train_fraction can not both be None!")

        self.epochs = epochs
        self.optimizer_cls = optimizer_cls
        self.lr = lr
        self.loss = loss

    def get_training_ids(self):
        # the constructor already checks if both or none are set
        all_patient_ids = self.data_handler.patient_ids

        if self.train_ids is not None:
            train_ids = self.train_ids
            # note: we can only use those patients from the training list
            # that are also contained in all_patients
            common = set(all_patient_ids).intersection(set(train_ids))
            train_ids = np.array(list(common))
            print("Have fixed train_ids")
        else:
            print("Choose random training fraction!")
            # get the training fraction and choose a random sample
            fraction = self.train_fraction

            train_ids = np.random.choice(
                all_patient_ids, replace=False,
                size=int(fraction * len(all_patient_ids)))

        assert set(train_ids).issubset(set(all_patient_ids))
        return np.array(sorted(train_ids))

    def create_compiled_model(self, input_shape):
        raise NotImplementedError(
            "You have to implement the create_compiled_model function!")

    def train_function(self, compiled_model, train_generator,
                       valid_generator, callbacks):
        """
        Execute model training

        Parameters
        ----------

        Returns
        -------
        a history object as returned from keras models
        """
        hist = compiled_model.fit_generator(
                train_generator,
                steps_per_epoch=np.ceil(
                    train_generator.n / train_generator.batch_size),
                epochs=self.epochs[0],
                verbose=1,
                validation_data=valid_generator,
                validation_steps=np.ceil(
                    valid_generator.n / valid_generator.batch_size),
                callbacks=callbacks
            )
        return hist

    def create_prediction_df(self, model, ids_train, ids_valid, ids_test,
                             avg_method="mean"):

        # TODO: handle the case that we have multiple outputs like
        # auxiliary outputs which have the same loss function but provide
        # different predictions
        # (i.e. if a cox loss is put after each dense layer)
        pred_df = model_predictions_per_patient(
            model, self.data_handler.img_dict,
            avg_method=avg_method,
            id_col=self.data_handler.id_col)

        # create a 'cohort' column
        cohort = [None] * len(pred_df)
        for i, (_, row) in enumerate(pred_df.iterrows()):
            pat = row[self.data_handler.id_col]
            if pat in ids_train:
                cohort[i] = "training"
            elif pat in ids_valid:
                cohort[i] = "validation"
            elif ids_test is not None and pat in ids_test:
                cohort[i] = "test"
            else:
                raise ValueError("Patient {} could not be assigned to a cohort!".format(
                    pat))
        pred_df.insert(loc=1, column="cohort", value=cohort)

        return pred_df

    def create_performance_df(self, pred_df):
        # TODO: handle the case that we have multiple outputs like
        # auxiliary outputs which have the same loss function but provide
        # different predictions
        # (i.e. if a cox loss is put after each dense layer)
        cis = compute_cis(
            pred_df, self.data_handler.labels,
            id_col=self.data_handler.id_col)

        # compute p values for stratification for training, validation and test (per slice and per patient)
        # based on the median risk of the training set
        pvals = compute_pvals(
            pred_df, self.data_handler.labels,
            id_col=self.data_handler.id_col)

        return pd.DataFrame({
            'train_ci_slice': cis['train_ci_slice'],
            'p_val_train_slice': pvals['train_p_slice'],

            'train_ci_pat': cis['train_ci_pat'],
            'p_val_train_pat': pvals['train_p_pat'],

            'valid_ci_slice': cis['valid_ci_slice'],
            'p_val_valid_slice': pvals['valid_p_slice'],

            'valid_ci_pat': cis['valid_ci_pat'],
            'p_val_valid_pat': pvals['valid_p_pat'],

            'test_ci_slice': cis['test_ci_slice'],
            'p_val_test_slice': pvals['test_p_slice'],

            'test_ci_pat': cis['test_ci_pat'],
            'p_val_test_pat': pvals['test_p_pat']}, index=[0])

    def evaluation_plots(self, model, pred_df, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        plot_km_and_scatter(
            pred_df, self.data_handler.labels,
            output_dir, id_col=self.data_handler.id_col)

    def evaluate_model(self, model, ids_train, ids_valid, ids_test,
                       output_dir):

        # make predictions for all patients
        pred_df = self.create_prediction_df(
            model, ids_train, ids_valid, ids_test)
        performance_df = self.create_performance_df(pred_df)

        # creation of Kaplan Meier and Scatter plots
        pred_plot_dirname = "predictions"
        output_dir = os.path.join(output_dir, pred_plot_dirname)
        self.evaluation_plots(model, pred_df, output_dir)

        # NOTE: we return preds as dict since other evaluation approaches
        # like autoencoders might have multiple sub-evaluations to perform
        # (i.e. for different number of chosen pca dimensions)
        return {pred_plot_dirname: (pred_df, performance_df)}


    def run_single_fold(self, train_ids, valid_ids, test_ids=None,
                        output_dir=None, print_model_summary=False,
                        rep=0, fold=0):

        self.data_handler.read_data_if_necessary()

        # arguments for data augmentation during training/validation
        train_generator, valid_generator = self.data_handler.create_generators(
            training_ids=train_ids,
            validation_ids=valid_ids)

        print("\nStarted rep={}, fold={}!\nTraining shape={},"
              " validation shape={}\n".format(
                rep, fold, train_generator.x.shape, valid_generator.x.shape))

        # create the model (already compiled!)
        input_shapes = [train_generator.x.shape[1:]]
        if (getattr(train_generator, "x_misc", None) and len(
           train_generator.x_misc) > 0):
            # we have additional inputs
            misc_shapes = [
                m.shape[1:] for m in train_generator.x_misc]
            input_shapes += misc_shapes

        if len(input_shapes) == 1:
            # for backward compatibility since most derived classes
            # expect only a single value as input_shape argument
            input_shapes = input_shapes[0]

        model = self.create_compiled_model(
            input_shape=input_shapes)
        if print_model_summary:
            print(model.summary(line_length=160))

        # define some useful callbacks for logging
        cb_csv = callbacks.CSVLogger(
            filename=os.path.join(output_dir, "training.csv"), append=True)
        # cb_tb = callbacks.TensorBoard(
        #     log_dir=os.path.join(fold_dir, "tf_log"),
        #     histogram_freq=5,
        #     batch_size=batch_size,
        #     write_graph=True,
        #     write_grads=True,
        #     write_images=True,
        #     update_freq='batch')
        cbs = [cb_csv]

        # train the model
        # from here users might want to have their individual training methods
        histories = self.train_function(
            model, train_generator, valid_generator,
            callbacks=cbs)
        if not isinstance(histories, collections.Iterable):
            histories = [histories]

        # store the model right after training
        model.save(os.path.join(output_dir, "trained_model.h5"))

        pred_and_perf = self.evaluate_model(
            model, train_ids, valid_ids, test_ids, output_dir=output_dir)

        # to the performance dfs we add the repetition and fold information
        # and the baseline results (if available).
        # this automatically adds a column with constant value of same length
        # as the given perf_df
        pred_dfs = {}
        perf_dfs = []
        for res_descr, (pred_df, perf_df) in pred_and_perf.items():
            # insert repetition and fold id into the performance dataframe
            perf_df.insert(loc=0, column="rep", value=rep)
            perf_df.insert(loc=1, column="fold", value=fold)

            perf_dfs.append(perf_df)

            # within the fold directory we create a new directory with the
            # results of the subevaluation
            sub_eval_dir = os.path.join(output_dir, res_descr)
            os.makedirs(sub_eval_dir, exist_ok=True)

            # store the repetition and the fold for the predictions
            # as well
            pred_df.insert(loc=1, column="rep", value=rep)
            pred_df.insert(loc=2, column="fold", value=fold)
            pred_df.to_csv(
                os.path.join(sub_eval_dir, "predictions.csv"),
                index=False)
            pred_dfs[res_descr] = pred_df

        # plotting of training histories
        plot_histories(
            histories=[("train_" + str(i+1), hist)
                       for i, hist in enumerate(histories)],
            keys=["loss", "ci"],
            save_dir=output_dir)
        plt.close()

        # combine all performances for this run from all
        # sub-evaluations by stacking rows
        full_perf_df = pd.concat(
            perf_dfs, ignore_index=True, sort=False)
        full_perf_df.to_csv(
            os.path.join(output_dir, "performance.csv"), index=False)

        return pred_dfs, full_perf_df

    def write_data_info(self, output_dir):
        """
        Allow to write additional information about e.g. the data
        if random crops were used or other stuff
        """
        pass
