import matplotlib
matplotlib.use("Agg")   # non interactive backend for not stopping computation for plotting

import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers, layers, Model
from keras import callbacks as keras_cb

from dl_toolbox.data.data_handler import DataHandler
from dl_toolbox.data.read import read_patient_data, read_baseline_feats
from dl_toolbox.preprocessing.preprocessing import stack_slices_for_patients
from dl_toolbox.models.common import dense_bn_act, dense_lrelu, conv_lrelu_2d
from dl_toolbox.cross_validation.run_cv_cmd import run_cv_from_cmd
from dl_toolbox.callbacks.concordance_index import ConcordanceIndex
from dl_toolbox.result_analysis.prediction_per_patient import model_predictions_per_patient
from dl_toolbox.cross_validation.cmd_line_parser import SurvivalCVCmdLineParser

from cv_train_from_scratch import FromScratchCVContext


class ClinicalParser(SurvivalCVCmdLineParser):
    def add_data_handler_args(self):
        super().add_data_handler_args()

        # now also add id_col, ...
        baseline_group = self._get_or_create_group("Baseline features")
        baseline_group.add_argument(
            "--baseline_feats", type=str, help="path to csv file containing baseline features for each patient.")
        baseline_group.add_argument(
            "--baseline_feat_cols", type=str, nargs="+", help="Column names for the features to use within <baseline_feats>")

    def get_data_handler_args(self):
        d = super().get_data_handler_args()

        d["baseline_feat_file"] = self.parsed_args.baseline_feats
        d["baseline_feat_cols"] = self.parsed_args.baseline_feat_cols

        return d


class DataHandlerMultiInput(DataHandler):
    """Uses baseline features as additional inputs to network"""
    def __init__(self,
                 input, outcome,
                 time_col,
                 event_col,
                 baseline_feat_file,
                 id_col="id",
                 baseline_feat_cols=None,
                 max_time_perturb=0,
                 batch_size=32,
                 mask_nontumor_areas=False,
                 no_data_augmentation=True,
                 training_augmentation_args={},
                 validation_augmentation_args={},
                 slices_around_max=(0, 0),
                 ):

        self.baseline_feat_file = baseline_feat_file
        self.baseline_feat_cols = baseline_feat_cols
        # will be filled when reading data
        self.baseline_feat_df = None

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

    def _read_data(self):

        data = read_patient_data(
            img_directories=self.input,
            outcome_file=self.outcome_file,
            time_col=self.time_col,
            event_col=self.event_col,
            id_col=self.id_col,
            n_around_max=self.n_around_max,
            preproc_fun=None)

        self.outcome_dict = data[0]
        self.img_dict = data[1]

        # check that we have outcome for all patients with images
        assert set(self.img_dict).issubset(set(self.outcome_dict))

        if self.mask_nontumor_areas:
            print("Will mask all nontumor areas with zeros!")  # Backup of img is found under key 'img_bk'.")
            for pat in self.img_dict:
                mask = self.img_dict[pat]["mask"]
                # img = self.img_dict[pat]["img"]
                # self.img_dict[pat]["img_bk"] = img
                self.img_dict[pat]["img"] *= mask

        # also read the baseline features
        baseline_feats = read_baseline_feats(
            self.baseline_feat_file, id_col=self.id_col)
        # limit the features to the given column names
        if self.baseline_feat_cols is None:
            # use all columns except for id
            print("No baseline_feat_cols specified, will use all!")
            self.baseline_feat_cols = [c for c in baseline_feats.columns if
                                       c != self.id_col]

        print("will use the following features: {}".format(
            self.baseline_feat_cols))

        self.baseline_feat_df = baseline_feats[
            [self.id_col] + self.baseline_feat_cols]

        # check that we have features for each patient with outcome
        # and vice versa and throw out others
        pats_with_outcome = np.array(sorted(self.outcome_dict.keys()))
        pats_with_features = np.array(sorted(
            pd.unique(self.baseline_feat_df[self.id_col])))
        pats_with_images = np.array(sorted(self.img_dict.keys()))

        complete_pats = sorted(list(set(pats_with_outcome).intersection(
            set(pats_with_features), set(pats_with_images))))

        print(f"{len(pats_with_outcome)} patients with outcome, {len(pats_with_features)} with features, with {len(pats_with_images)} images and {len(complete_pats)} with all.")
        self.baseline_feat_df = self.baseline_feat_df[
            self.baseline_feat_df[self.id_col].isin(complete_pats)]

        self.outcome_dict = {p: self.outcome_dict[p] for p in complete_pats}
        self.img_dict = {p: self.img_dict[p] for p in complete_pats}

        print("Full data remaining:", self.baseline_feat_df.shape, len(self.outcome_dict), len(self.img_dict))
        self.patient_ids = complete_pats

    def stacked_np_array(self, patient_ids=None, slice_idx_key="slice_idx"):
        if patient_ids is None:
            patient_ids = self.patient_ids

        # check that the ids are present in outcome and img data
        if not set(patient_ids).issubset(set(self.patient_ids)):
            raise ValueError(
                "No data available for requested patients {}".format(
                    set(patient_ids) - set(self.patient_ids)))

        # also check that the ids are within the baseline feature df
        baseline_data_ids = self.baseline_feat_df[self.id_col].values
        if not set(patient_ids).issubset(set(baseline_data_ids)):
            raise ValueError(
                "No baseline features available for {}".format(
                    set(patient_ids) - set(baseline_data_ids)))

        return stack_slices_for_patients(
                self.img_dict, self.labels, patient_ids,
                max_time_perturb=self.max_time_perturb,
                slice_idx_key=slice_idx_key,
                additional_feature_df=self.baseline_feat_df,
                id_col=self.id_col)

    def create_generators(self, training_ids, validation_ids):

        train_imgs, _, train_labels, _, train_features = self.stacked_np_array(
            training_ids)
        # print(train_imgs.shape, train_features.shape, train_labels.shape)
        valid_imgs, _, valid_labels, _, valid_features = self.stacked_np_array(
            validation_ids)
        # print(valid_imgs.shape, valid_features.shape, valid_labels.shape)

        if self.no_data_augmentation:
            print("No data augmentation will be done!")
            train_datagen = ImageDataGenerator()
            valid_datagen = ImageDataGenerator()
        else:
            train_datagen = ImageDataGenerator(
                **self.training_augmentation_args)
            valid_datagen = ImageDataGenerator(
                **self.validation_augmentation_args)

        train_generator = train_datagen.flow(
            (train_imgs, train_features), train_labels,
            batch_size=self.batch_size, shuffle=True)

        valid_generator = valid_datagen.flow(
            (valid_imgs, valid_features), valid_labels,
            batch_size=self.batch_size, shuffle=False)

        return train_generator, valid_generator


class FromScratchMultiInputCVContext(FromScratchCVContext):
    def create_compiled_model(self, input_shape):
        """This will now get an image input and an input for other features"""

        assert len(input_shape) >= 2

        # this has only an image input
        m_img = super().create_compiled_model(input_shape[0])
        x = m_img.output

        args = {'kernel_regularizer': regularizers.l1_l2(
            self.l1, self.l2)}
        # now use the remaining data as additional input
        other_in = layers.Input(input_shape[1])
        y = dense_bn_act(other_in, 1, activation=self.finalact,
                         bn=self.batchnorm, dense_args=args,
                         name="other_out")

        # concatenate the estimated "risks" for both inputs
        # and out of those create a final risk that is free
        # to weight the two risk features as it sees fit
        z = layers.Concatenate()([x, y])
        z = dense_bn_act(z, 1, activation=self.finalact,
                         bn=self.batchnorm, dense_args=args,
                         name="combined_out")

        m = Model([m_img.input, other_in], z)
        m.compile(
            loss=m_img.loss,
            optimizer=m_img.optimizer)

        return m

    def train_function(self, compiled_model, train_generator,
                       valid_generator, callbacks):

        early_stop = keras_cb.EarlyStopping(
            monitor="val_loss", patience=10, verbose=1, mode="min")
        reduce_lr = keras_cb.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1, mode="min",
            min_lr=1e-7)
        cb_ci = ConcordanceIndex(
            train_data=([train_generator.x, train_generator.x_misc[0]],
                        train_generator.y),
            valid_data=([valid_generator.x, valid_generator.x_misc[0]],
                        valid_generator.y),
            freq=1)

        callbacks = [
            early_stop, reduce_lr,
            cb_ci] + callbacks

        hist = compiled_model.fit_generator(
            train_generator,
            steps_per_epoch=np.ceil(
                train_generator.n / train_generator.batch_size),
            epochs=self.epochs[0],
            verbose=1,
            validation_data=valid_generator,
            validation_steps=np.ceil(
                valid_generator.n / valid_generator.batch_size),
            callbacks=callbacks)

        return hist

    # this needs tweaking of the prediction function
    # as well since we need additional inputs
    def create_prediction_df(self, model, ids_train, ids_valid, ids_test,
                             avg_method="mean"):
        pred_df = model_predictions_per_patient(
            model, self.data_handler.img_dict,
            avg_method=avg_method,
            additional_feature_df=self.data_handler.baseline_feat_df,
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
                raise ValueError(
                    f"Patient {pat} could not be assigned to a cohort!")
        pred_df.insert(loc=1, column="cohort", value=cohort)

        return pred_df


if __name__ == "__main__":
    run_cv_from_cmd(
        cv_context_cls=FromScratchMultiInputCVContext,
        data_handler_cls=DataHandlerMultiInput,
        parser_cls=ClinicalParser,
        ensemble_method=np.mean)
