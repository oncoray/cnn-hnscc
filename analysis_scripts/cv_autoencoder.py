import matplotlib
matplotlib.use("Agg")   # non interactive backend for not stopping computation for plotting

import argparse
import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

from lifelines import CoxPHFitter

from keras import backend as K
from keras import Model, optimizers
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from keras.layers import Input, MaxPooling2D, Flatten, Reshape, UpSampling2D,\
    Conv2D

from sklearn.decomposition import PCA
from joblib import dump  # for storing the sklearn models via pickle

from dl_toolbox.models.common import conv_lrelu_2d, dense_lrelu, dense_bn_act, conv_bn_act
from dl_toolbox.callbacks import ConcordanceIndex
from dl_toolbox.result_analysis.evaluate_model import compute_cis,\
    compute_pvals, plot_km_and_scatter

from dl_toolbox.cross_validation.run_cv_cmd import run_cv_from_cmd

from dl_toolbox.data.data_handler import DataHandler
from dl_toolbox.cross_validation.cmd_line_parser import SurvivalCVCmdLineParser

from dl_feature_extraction import extract_features_flattened,\
    extract_feature_info

from cv_train_from_scratch import FromScratchCVContext


class AutoencoderParser(SurvivalCVCmdLineParser):

    def add_context_args(self):
        super().add_context_args()

        # now add the pca group to
        pca_group = self._get_or_create_group("PCA")
        pca_group.add_argument(
            "--pca_dims", type=int, default=3, nargs="+",
            help="number of pca dimensions to reduce encoding features to")

    def get_context_args(self):
        args = super().get_context_args()

        args["pca_dims"] = self.parsed_args.pca_dims

        return args


def _encoder(input_shape, n_layers, n_filters_start, bn, lrelu, conv_args,
             bottleneck_filters):
    inputs = Input(input_shape)
    x = inputs
    for i in range(n_layers):
        n_filters = n_filters_start * 2**i
        name = "conv_" + str(i+1)
        x = conv_lrelu_2d(x, n_filters, strides=1, filter_size=3,
                          lrelu=lrelu, bn=bn, conv_args=conv_args,
                          name=name+"_1")
        #x = conv_lrelu_2d(x, n_filters, strides=1, filter_size=3,
        #                  lrelu=lrelu, bn=bn, conv_args=conv_args, name=name+"_2")
        if i < n_layers - 1:
            x = MaxPooling2D()(x)

    # now further reduce the number of bottleneck features by doing another 1x1 conv with fewer
    # filters
    x = conv_lrelu_2d(x, n_filters=bottleneck_filters,
                      strides=1, filter_size=3, lrelu=lrelu, bn=bn,
                      conv_args=conv_args, name="bottleneck")

    return Model(inputs=inputs, outputs=x, name="Encoder")


def _decoder(input_shape, n_layers, n_filters_start, bn, lrelu, conv_args):
    inputs = Input(input_shape)
    x = inputs

    for i in range(n_layers):
        n_filters = int(n_filters_start * 2**(-i))
        name = "upconv_" + str(i+1)
        x = conv_lrelu_2d(x, n_filters=n_filters, strides=1,
                          filter_size=3, lrelu=lrelu, bn=bn,
                          conv_args=conv_args, name=name+"_1")
        if i < n_layers - 1:
            # use upsampling instead of transposed convolution to avoid
            # checkerboard patterns
            x = UpSampling2D()(x)

    x = conv_bn_act(x, Conv2D, 1, filter_size=1, strides=1,
                    activation="sigmoid", bn=bn, conv_args=conv_args,
                    name="output")

    return Model(inputs, x, name="Decoder")


# needs to create generators that do not serve survival data but
# rather the images as inputs and outputs

class AutoencoderGenerator(NumpyArrayIterator):
    def __init__(self, x, augmentation_args={}, batch_size=32, shuffle=True):

        super().__init__(
            x=x, y=None,
            image_data_generator=ImageDataGenerator(**augmentation_args),
            batch_size=batch_size,
            shuffle=shuffle)

    def _get_batches_of_transformed_samples(self, index_array):
        # now only return the image
        batch_x = super()._get_batches_of_transformed_samples(index_array)

        return batch_x, batch_x


class AutoencoderDataHandler(DataHandler):

    def create_generators(self, training_ids, validation_ids):

        train_imgs, _, _, _ = self.stacked_np_array(training_ids)
        valid_imgs, _, _, _ = self.stacked_np_array(validation_ids)

        # NOTE: the train_generator spits out survival labels, not the images
        # themselves so we have to tweak it to get a generator that uses images
        # as inputs and as outputs (and also applies same transformations)

        train_gen = AutoencoderGenerator(
            train_imgs, self.training_augmentation_args, self.batch_size,
            shuffle=True)
        valid_gen = AutoencoderGenerator(
            valid_imgs, self.validation_augmentation_args, self.batch_size,
            shuffle=False)

        return train_gen, valid_gen


def plot_recon_for_random_samples(model, images, slice_ids,
                                  n=1, save_dir=None, seed=1):
    np.random.seed(seed)
    vis_sample_idx = np.random.choice(
        range(len(images)), n, replace=False)
    vis_samples = images[vis_sample_idx]
    vis_reco = model.predict(vis_samples)

    vis_slice_ids = slice_ids[vis_sample_idx]

    f, axs = plt.subplots(n, 2, figsize=(10, 5*n))
    axs = axs.reshape((n, 2))
    for i in range(n):
        axs[i, 0].imshow(vis_samples[i].squeeze(), cmap="gray")
        axs[i, 0].set_title("CT {}".format(vis_slice_ids[i]))

        axs[i, 1].imshow(vis_reco[i].squeeze(), cmap="gray")
        axs[i, 1].set_title("Autoencoder reconstruction of {}".format(
            vis_slice_ids[i]))

    if save_dir is not None:
        plt.savefig(save_dir)
        print("saved plot to", save_dir)


def create_feature_df(model, imgs, labels, slice_descriptions,
                      ids_train, id_col, time_col, event_col):

    info_df = extract_feature_info(
        labels, slice_descriptions, ids_train,
        id_col=id_col,
        time_col=time_col,
        event_col=event_col)
    feature_df, _ = extract_features_flattened(model, imgs)

    return pd.concat([info_df, feature_df], axis=1)


def apply_pca_transform(pca, df, feat_cols, output_path):
    pca_feat = pca.transform(df[feat_cols].values)
    # convert to dataframe
    pca_feat = pd.DataFrame(pca_feat, columns=[
            "feat_" + str(i+1) for i in range(pca_feat.shape[1])])

    # store the pca features and their info
    info_cols = [c for c in df.columns if c not in feat_cols]
    pca_df = pd.concat([df[info_cols], pca_feat], axis=1)

    pca_df.to_csv(output_path, index=False)
    print("Stored PCA features to", output_path)

    return pca_df


class AutoencoderCVContext(FromScratchCVContext):

    def __init__(self,
                 data_handler,
                 seed=1,
                 train_ids=None,
                 train_fraction=None,
                 epochs=[10],
                 optimizer_cls=optimizers.Adam,
                 lr=[1.e-3],
                 loss="binary_crossentropy",
                 #### up to here base class args
                 batchnorm=None,
                 lrelu=0.01,
                 l1=0.,
                 l2=0.,
                 dropout=0.,
                 finalact="sigmoid",
                 #
                 n_layers=6,
                 n_filters_start=16,
                 n_bottleneck_filters=64,
                 pca_dims=[1, 2, 5, 10],
                 **kwargs):

        self.n_layers = n_layers
        self.n_filters_start = n_filters_start
        self.n_bottleneck_filters = n_bottleneck_filters
        self.pca_dims = pca_dims

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
        conv_args = dict(
            padding="same", kernel_initializer="he_normal",
            bias_initializer="random_normal", kernel_regularizer=None)
        # dense_args = dict(
        #     kernel_initializer="he_normal", bias_initializer="random_normal",
        #     kernel_regularizer=None)

        # so the bottleneck shape will be
        # (input_shape / 2**(n_layers-1))**2 * bottleneck_filters

        inputs = Input(input_shape)
        enc = _encoder(
            input_shape, self.n_layers, self.n_filters_start, self.batchnorm,
            self.lrelu, conv_args, self.n_bottleneck_filters)

        dec_n_start = int(self.n_filters_start * 2**(self.n_layers-1))
        dec = _decoder(K.int_shape(enc.get_output_at(0))[1:],
                       self.n_layers, dec_n_start, self.batchnorm,
                       self.lrelu, conv_args)

        autoencoder = Model(
            inputs=inputs, outputs=dec(enc(inputs)), name="Autoencoder")

        autoencoder.compile(
            loss=self.loss,
            optimizer=self.optimizer_cls(lr=self.lr[0]),
            metrics=["mse", "mae"],
        )
        return autoencoder

    def _reconstruction_plots(self, model, ids_train, ids_valid, ids_test,
                              output_dir):
        cohorts = ["training", "validation", "test"]
        for cohort, ids in zip(cohorts, [ids_train, ids_valid, ids_test]):
            if ids is None:
                continue
            imgs, _, _, ids_stacked = self.data_handler.stacked_np_array(
                ids)
            plot_recon_for_random_samples(
                model, imgs, ids_stacked, n=1,
                save_dir=os.path.join(
                    output_dir, "autoencoder_recon_{}.png".format(
                        cohort)),
                seed=1)
            plt.close()

    def _get_encoder_features_as_df(self, model, ids_train, ids_valid,
                                    ids_test, output_dir):
        # predict the encoding part and write those features
        # to disk (reuse function of dl_feature_extraction if possible)
        # so we can later run the FAMILIAR framework on the encoder features
        # with Lasso-Cox (not possible with lifelines which only does standard-cox)
        encoder = model.get_layer("Encoder")

        cohorts = ["training", "validation", "test"]
        cohort_dfs = {c: None for c in cohorts}
        for cohort, ids in zip(cohorts, [ids_train, ids_valid, ids_test]):
            if ids is None:
                continue

            imgs, _, labels, ids_stack = self.data_handler.stacked_np_array(
                ids)
            # now we need to predict and flatten and store the
            # encoder features
            df = create_feature_df(
                encoder, imgs, labels, ids_stack, ids_train,
                id_col=self.data_handler.id_col,
                time_col=self.data_handler.time_col,
                event_col=self.data_handler.event_col)

            if cohort == "validation":
                # note the validation ids will also be marked as 'training'
                # in the csv file (since they are basically part of the training set
                # during crossvalidation but here we want it to say validation
                df.cohort.replace({"training": "validation"}, inplace=True)

            df.to_csv(
                os.path.join(output_dir, "encoder_features_{}.csv".format(
                    cohort)),
                index=False)

            cohort_dfs[cohort] = df

        return cohort_dfs

    def train_function(self, compiled_model, train_generator,
                       valid_generator, callbacks):

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

    def evaluate_model(self, model, ids_train, ids_valid, ids_test,
                       output_dir):

        self._reconstruction_plots(
            model, ids_train, ids_valid, ids_test,
            output_dir=output_dir)

        cohort_dfs_encoder = self._get_encoder_features_as_df(
            model, ids_train, ids_valid, ids_test, output_dir)

        train_df = cohort_dfs_encoder["training"]

        # Now do PCA and apply normal cox models
        feat_cols = [c for c in train_df.columns if c.startswith("feat_")]
        # info_cols = [c for c in train_df.columns if c not in feat_cols]

        ret = dict()
        for pca_dim in self.pca_dims:
            pca = PCA(n_components=pca_dim)
            pca.fit(train_df[feat_cols].values)

            # apply pca transformations on all sets for further
            # dimensionality reduction
            pca_dfs = dict()
            for name, df in cohort_dfs_encoder.items():
                output_path = os.path.join(
                    output_dir, "pca_{}comp_features_{}.csv".format(
                        pca_dim, name))
                pca_dfs[name] = apply_pca_transform(
                    pca, df, feat_cols, output_path)

            # now we can combine the datasets to a huge one
            # containing pca features for training, validation and test patients
            pca_df_concat = pd.concat(
                list(pca_dfs.values()), axis=0, sort=False)

            # evaluate the concordance index of Cox models that use
            # PCA reduced features of the autoencoder
            print("\nPCA with {} components\n".format(pca_dim))
            drop_cols = [self.data_handler.id_col, "slice_idx", "cohort"]  # only survival info is left
            cox_fitter = CoxPHFitter()
            try:
                cox_fitter.fit(
                    pca_dfs["training"].drop(drop_cols, axis=1),
                    duration_col=self.data_handler.time_col,
                    event_col=self.data_handler.event_col,
                    show_progress=False)
                cox_fitter.print_summary()
            except Exception as e:
                print("[W]: Fitting cox model failed! Reason: {}".format(e))
                continue

            # now create the prediction dataframe that we can then use
            # for computing ci and pvalues easily
            id_col = self.data_handler.id_col
            ids = np.unique(pca_df_concat[id_col].values)
            cohort = [None] * len(ids)
            slice_idx = [None] * len(ids)
            pred_risk_per_slice = [None] * len(ids)
            for i, pat in enumerate(ids):
                # find all slices for that patient
                if pat in ids_train:
                    cohort[i] = "training"
                elif pat in ids_valid:
                    cohort[i] = "validation"
                elif ids_test is not None and pat in ids_test:
                    cohort[i] = "test"
                else:
                    msg = "Patient {} could not be assigned to a cohort!".format(
                        pat)
                    raise ValueError(msg)

                pat_df = pca_df_concat[pca_df_concat[id_col] == pat]
                haz = cox_fitter.predict_log_partial_hazard(
                    pat_df.drop(drop_cols, axis=1))

                hazard = haz.values.flatten()
                slice_idx[i] = pat_df.slice_idx.values.tolist()
                pred_risk_per_slice[i] = hazard

            pred_df = pd.DataFrame({
                id_col: ids,
                'cohort': cohort,
                'slice_idx': slice_idx,
                'pred_per_slice': pred_risk_per_slice,
                'pred_per_pat(mean)': [
                    np.mean(slice_preds) for slice_preds in pred_risk_per_slice],
                'pred_variance': [
                    np.var(slice_preds) for slice_preds in pred_risk_per_slice]
            })

            cis = compute_cis(
                pred_df, self.data_handler.outcome_dict,
                id_col=id_col)
            pvals = compute_pvals(
                pred_df, self.data_handler.outcome_dict,
                id_col=id_col)

            performance_df = pd.DataFrame({
                'pca_dim': [pca_dim],
                'pca_explained_variance': [
                    pca.explained_variance_ratio_.tolist()],

                'train_ci_slice': [cis['train_ci_slice']],
                'p_val_train_slice': [pvals['train_p_slice']],

                'train_ci_pat': [cis['train_ci_pat']],
                'p_val_train_pat': [pvals['train_p_pat']],

                'valid_ci_slice': [cis['valid_ci_slice']],
                'p_val_valid_slice': [pvals['valid_p_slice']],

                'valid_ci_pat': [cis['valid_ci_pat']],
                'p_val_valid_pat': [pvals['valid_p_pat']],

                'test_ci_slice': [cis['test_ci_slice']],
                'p_val_test_slice': [pvals['test_p_slice']],

                'test_ci_pat': [cis['test_ci_pat']],
                'p_val_test_pat': [pvals['test_p_pat']]})

            subexp_name = "predictions_pca_"+str(pca_dim)+"_comp"
            ret[subexp_name] = (pred_df, performance_df)

            subexp_path = os.path.join(output_dir, subexp_name)
            os.makedirs(subexp_path, exist_ok=True)
            # kaplan meier and risk_vs_survival plots!
            plot_km_and_scatter(
                pred_df, self.data_handler.outcome_dict,
                output_dir=subexp_path,
                id_col=id_col)

            # save the transformation matrix V and the training mean
            # such that pca_train = (enc_train-mean(enc_train)) * V.T
            # and we can later work with those models
            dump(pca, os.path.join(
                subexp_path, "PCA_" + str(pca_dim) + "comp.joblib"))
            cox_fitter.summary.to_csv(
                os.path.join(
                    subexp_path, "cox_{}_pca-comp_summary.csv".format(
                        pca_dim)),
                index=False)
            cox_fitter.params_.to_csv(
                os.path.join(
                    subexp_path, "cox_{}_pca-comp_coefs.csv".format(
                        pca_dim)),
                index=False)

        # we return a tuple of prediction_df, performance_df
        # for each run
        # of the PCA with different dimensionality
        return ret


if __name__ == "__main__":
    run_cv_from_cmd(
        cv_context_cls=AutoencoderCVContext,
        data_handler_cls=AutoencoderDataHandler,
        parser_cls=AutoencoderParser,
        ensemble_method=np.mean
    )
