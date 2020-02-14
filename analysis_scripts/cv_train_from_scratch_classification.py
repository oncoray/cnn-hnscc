import matplotlib
matplotlib.use("Agg")   # non interactive backend for not stopping computation for plotting

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

from functools import partial

from keras import layers, Model, regularizers, optimizers
from keras import callbacks as keras_cb

from dl_toolbox.cross_validation.run_cv_cmd import run_cv_from_cmd
from dl_toolbox.cross_validation.ensemble import evaluate_ensemble_auc
from dl_toolbox.models.common import dense_bn_act, dense_lrelu, conv_lrelu_2d
from dl_toolbox.data.data_handler import DataHandler
from dl_toolbox.result_analysis.evaluate_model import preds_and_labels_per_slice_and_pat,\
    _get_preds_and_labels_per_slice_and_patient, compute_pval
from dl_toolbox.visualization import plot_stratified_cohort_km, plot_auc_curves, plot_confusion_matrices

from sklearn.metrics import roc_auc_score, classification_report,\
    precision_recall_fscore_support
from sklearn.utils import class_weight

from cv_train_from_scratch import FromScratchCVContext


class DataHandlerClassification(DataHandler):

    def __init__(self,
                 input, outcome,
                 time_col,
                 event_col,
                 id_col="id",
                 max_time_perturb=0,
                 batch_size=32,
                 mask_nontumor_areas=False,
                 no_data_augmentation=True,
                 training_augmentation_args={},
                 validation_augmentation_args={},
                 slices_around_max=(0, 0),
                 survival_cutoff=24):

        # the cutoff for binarizing the endpoint
        self.survival_cutoff = survival_cutoff

        super().__init__(
            input=input, outcome=outcome,
            time_col=time_col, event_col=event_col, id_col=id_col,
            max_time_perturb=max_time_perturb,
            batch_size=batch_size,
            mask_nontumor_areas=mask_nontumor_areas,
            no_data_augmentation=no_data_augmentation,
            training_augmentation_args=training_augmentation_args,
            validation_augmentation_args=validation_augmentation_args,
            slices_around_max=slices_around_max)

    def _make_label_dict_from_outcomes(self):
        # alter the outcome dict to contain only classification information
        # now tweak the labels from a 2D (time, censoring) format to
        # a classification format
        # 0 means did not survive up to the survival_cutoff (i.e. high risk)
        # and 1 means survived at least up to the cutoff (i.e. low risk)
        labels = {}
        for pat, (time, _) in self.outcome_dict.items():
            labels[pat] = int(time >= self.survival_cutoff)

        # print("Data handler: labels = {}".format(labels))
        return labels

    def _read_data(self):
        super()._read_data()

        # now filter the outcome dict, the img_dict, the baseline feat df
        # and the patient ids
        # so that patients are excluded when they are censored before the cutoff
        # since we don't know about their binary survival
        exclude_ids = [pat for pat, (time, event) in self.outcome_dict.items()
                       if event == 0 and time < self.survival_cutoff]

        for pat in exclude_ids:
            print("Remove {} from classification data since censored before cutoff {}(label = {})".format(
                pat, self.survival_cutoff, self.outcome_dict[pat]))

            del self.img_dict[pat]
            del self.outcome_dict[pat]

        print("Excluded {} patients due to survival cutoff.".format(
            len(exclude_ids)))

        self.patient_ids = list(self.outcome_dict.keys())


class FromScratchCVContextClassification(FromScratchCVContext):
    def __init__(self,
                 data_handler,
                 seed=1,
                 train_ids=None,
                 train_fraction=.75,
                 epochs=[10],
                 optimizer_cls=optimizers.Adam,
                 lr=[1.e-3],
                 loss="binary_crossentropy",
                 #### up to here base class args
                 batchnorm=None,
                 lrelu=0.,
                 l1=0.,
                 l2=0.,
                 dropout=0.,
                 finalact="sigmoid",
                 ####
                 class_thresholds=[0.5],
                 **kwargs):

        # for converting neural network output to a class label
        self.class_thresholds = class_thresholds
        # for creation of class labels from survival time labels
        # this raises Exception if the data handler does not have that
        # attribute
        self.survival_cutoff = getattr(data_handler, "survival_cutoff")

        super().__init__(
            data_handler,
            seed=seed,
            train_ids=train_ids,
            train_fraction=train_fraction,
            epochs=epochs,
            optimizer_cls=optimizer_cls,
            lr=lr,
            loss=loss,
            batchnorm=batchnorm,
            lrelu=lrelu,
            l1=l1,
            l2=l2,
            dropout=dropout,
            finalact=finalact,
            **kwargs)

    def create_compiled_model(self, input_shape):

        m = super().create_compiled_model(input_shape)
        # re-compile to also get accuracy metric
        m.compile(
            loss=m.loss,
            optimizer=m.optimizer,
            metrics=["accuracy"])

        return m

    def train_function(self, compiled_model, train_generator,
                       valid_generator, callbacks):

        early_stop = keras_cb.EarlyStopping(
            monitor="val_loss", patience=10, verbose=1, mode="min")
        reduce_lr = keras_cb.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1, mode="min",
            min_lr=1e-7)

        callbacks = [early_stop, reduce_lr] + callbacks

        y_train = train_generator.y.squeeze()
        class_weights = class_weight.compute_class_weight(
            'balanced', np.unique(y_train), y_train)

        class_weights = dict(enumerate(class_weights))
        class_1 = np.sum(y_train)
        class_0 = len(y_train) - class_1
        print("samples class 0: {}, samples class 1: {}".format(
            class_0, class_1))
        print("class_weights are", class_weights)

        hist = compiled_model.fit_generator(
                train_generator,
                steps_per_epoch=np.ceil(
                    train_generator.n / train_generator.batch_size),
                epochs=self.epochs[0],
                verbose=1,
                validation_data=valid_generator,
                validation_steps=np.ceil(
                    valid_generator.n / valid_generator.batch_size),
                callbacks=callbacks,
                class_weight=class_weights)
        return hist

    def create_performance_df(self, pred_df):
        # compute the AUC, precision, recall and f1-measure
        aucs = {}
        prec = {}
        recall = {}
        fscore = {}
        pval = {}
        cohorts = np.unique(pred_df.cohort.values)
        for cohort in cohorts:
            cohort_df = pred_df[pred_df.cohort == cohort]
            ids = cohort_df[self.data_handler.id_col].values

            pred_slice, labels_slice, pred_pat, labels_pat = preds_and_labels_per_slice_and_pat(
                cohort_df, self.data_handler.labels, ids,
                self.data_handler.id_col)

            # print(cohort, pred_slice.shape, labels_slice.shape, pred_pat.shape, labels_pat.shape)
            # print("pred_slice", pred_slice)
            auc_slice = roc_auc_score(labels_slice, pred_slice)
            auc_pat = roc_auc_score(labels_pat, pred_pat)

            aucs[cohort] = (auc_slice, auc_pat)

            # use precision_recall_fscore_support function to append the metrics
            # to the dataframe
            # but to do this we have to convert predictions (result of sigmoid layer)
            # to class labels based on the given class thresholds
            pred_cls_slice = np.digitize(pred_slice, self.class_thresholds)
            pred_cls_pat = np.digitize(pred_pat, self.class_thresholds)

            prec_slice, recall_slice, f1_slice, _ = precision_recall_fscore_support(
                labels_slice, pred_cls_slice)

            prec_pat, recall_pat, f1_pat, _ = precision_recall_fscore_support(
                labels_pat, pred_cls_pat)

            prec[cohort] = (prec_slice, prec_pat)
            recall[cohort] = (recall_slice, recall_pat)
            fscore[cohort] = (f1_slice, f1_pat)

            # p value of predicted stratification
            # but we need the full survival label, not the binary label
            _, surv_labels_slice, _, surv_labels_pat = preds_and_labels_per_slice_and_pat(
                cohort_df, self.data_handler.outcome_dict, ids,
                self.data_handler.id_col)

            pval_slice = compute_pval(
                pred_cls_slice, surv_labels_slice, threshold=0, alpha=0.95)
            pval_pat = compute_pval(
                pred_cls_pat, surv_labels_pat, threshold=0, alpha=0.95)
            pval[cohort] = (pval_slice, pval_pat)

        metrics = zip(["AUC", "precision", "recall", "F1_score", "p_val"],
                      [aucs, prec, recall, fscore, pval])
        metrics_df = []
        for metric_str, metric in metrics:
            df = pd.DataFrame({
                'train_{}_slice'.format(metric_str): [metric["training"][0]],
                'train_{}_pat'.format(metric_str): [metric["training"][1]],

                'valid_{}_slice'.format(metric_str): [metric["validation"][0]],
                'valid_{}_pat'.format(metric_str): [metric["validation"][1]],

                'test_{}_slice'.format(metric_str): [metric["test"][0]],
                'test_{}_pat'.format(metric_str): [metric["test"][1]]
            })
            metrics_df.append(df)

        # concatenate the columns of the different metrics
        print("\nPER SLICE AUC: train: {}, valid: {}, test: {}".format(
            aucs["training"][0], aucs["validation"][0], aucs["test"][0]))

        print("PER PATIENT AUC: train: {}, valid: {}, test: {}".format(
            aucs["training"][1], aucs["validation"][1], aucs["test"][1]))

        return pd.concat(metrics_df, axis=1)

    def evaluation_plots(self, model, pred_df, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        # NOTE: the labels we use here are still time and outcome, not binary
        # so all labels have to be converted if evaluated on classification
        train, valid, test = _get_preds_and_labels_per_slice_and_patient(
                pred_df, self.data_handler.outcome_dict,
                self.data_handler.id_col)

        tr_p_slice, tr_l_slice, tr_p_pat, tr_l_pat = train
        v_p_slice, v_l_slice, v_p_pat, v_l_pat = valid
        te_p_slice, te_l_slice, te_p_pat, te_l_pat = test
        print("in evaluation_plots, shapes are", tr_p_slice.shape, tr_l_slice.shape)

        # convert model prediction (sigmoid output) to class predictions (hard labels)
        tr_class_slice = np.digitize(tr_p_slice, self.class_thresholds)
        tr_class_pat = np.digitize(tr_p_pat, self.class_thresholds)

        v_class_slice = np.digitize(v_p_slice, self.class_thresholds)
        v_class_pat = np.digitize(v_p_pat, self.class_thresholds)

        te_class_slice = np.digitize(te_p_slice, self.class_thresholds)
        te_class_pat = np.digitize(te_p_pat, self.class_thresholds)

        # label times also have to be converted to 0/1 class for AUC evaluation
        tr_l_slice_binary = (tr_l_slice[:, 0] >= self.survival_cutoff).astype(int)
        tr_l_pat_binary = (tr_l_pat[:, 0] >= self.survival_cutoff).astype(int)

        v_l_slice_binary = (v_l_slice[:, 0] >= self.survival_cutoff).astype(int)
        v_l_pat_binary = (v_l_pat[:, 0] >= self.survival_cutoff).astype(int)

        te_l_slice_binary = (te_l_slice[:, 0] >= self.survival_cutoff).astype(int)
        te_l_pat_binary = (te_l_pat[:, 0] >= self.survival_cutoff).astype(int)


        # plot AUC curves for training, validation and test (per slice and per patient)
        plot_auc_curves(
            tr_p_slice, tr_l_slice_binary,
            v_p_slice, v_l_slice_binary,
            te_p_slice, te_l_slice_binary,
            subtitle="(per slice)",
            save_dir=os.path.join(output_dir, "auc_per_slice.png"))
        plt.close()

        plot_auc_curves(
            tr_p_pat, tr_l_pat_binary,
            v_p_pat, v_l_pat_binary,
            te_p_pat, te_l_pat_binary,
            subtitle="(per patient)",
            save_dir=os.path.join(output_dir, "auc_per_patient.png"))
        plt.close()

        # plot confusion matrices for training, validation and test
        plot_confusion_matrices(
            tr_class_slice, tr_l_slice_binary,
            v_class_slice, v_l_slice_binary,
            pred_test=te_class_slice, test_labels=te_l_slice_binary,
            subtitle="(per slice)",
            save_dir=os.path.join(output_dir, "confusion_matrices_per_slice_pred_cutoff={}.png".format(
                str(self.class_thresholds))))
        plt.close()

        plot_confusion_matrices(
            tr_class_pat, tr_l_pat_binary,
            v_class_pat, v_l_pat_binary,
            pred_test=te_class_pat, test_labels=te_l_pat_binary,
            subtitle="(per patient)",
            save_dir=os.path.join(output_dir, "confusion_matrices_per_patient_pred_cutoff={}.png".format(
                str(self.class_thresholds))))
        plt.close()

        # plot kaplan meier curves for the classes that were predicted
        # based on per slice and per patient
        # this needs true event times and labels
        f, axs = plt.subplots(1, 3, figsize=(12,4))
        plot_stratified_cohort_km(
            axs,
            condition=lambda x: x == 0,  # label 0 -> time < self.cutoff -> high risk
            pred_train=tr_class_slice, train_labels=tr_l_slice,
            pred_valid=v_class_slice, valid_labels=v_l_slice,
            pred_test=te_class_slice, test_labels=te_l_slice,
            subtitle=None,
            save_dir=os.path.join(output_dir, "kaplan_meier_of_predicted_classes_per_slice_cutoff={}.png".format(
                str(self.class_thresholds))),
            strata_labels=["High risk", "Low risk"],
            y_label="Loco-regional recurrence")

        plt.close()

        f, axs = plt.subplots(1, 3, figsize=(12,4))
        plot_stratified_cohort_km(
            axs,
            condition=lambda x: x == 0,  # label 0 -> time < self.cutoff -> high risk
            pred_train=tr_class_pat, train_labels=tr_l_pat,
            pred_valid=v_class_pat, valid_labels=v_l_pat,
            pred_test=te_class_pat, test_labels=te_l_pat,
            subtitle=None,
            save_dir=os.path.join(output_dir, "kaplan_meier_of_predicted_classes_per_patient_cutoff={}.png".format(
                str(self.class_thresholds))),
            strata_labels=["High risk", "Low risk"],
            y_label="Loco-regional recurrence")

        plt.close()


if __name__ == "__main__":

    run_cv_from_cmd(
        cv_context_cls=FromScratchCVContextClassification,
        data_handler_cls=partial(
            DataHandlerClassification, survival_cutoff=24),
        ensemble_method=np.mean,
        ensemble_metric_fns=[evaluate_ensemble_auc])
