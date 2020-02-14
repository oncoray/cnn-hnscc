import matplotlib
matplotlib.use("Agg")   # non interactive backend for not stopping computation for plotting

# import matplotlib.pyplot as plt

import argparse
import numpy as np

from keras import backend as K
from keras import callbacks as keras_cb
from keras.applications import resnet50, densenet, inception_resnet_v2, vgg16, inception_v3
from keras import regularizers, optimizers

from dl_toolbox.models import pretrained_model, get_base_model
from dl_toolbox.preprocessing import normalize_img, to_rgb
from dl_toolbox.losses import neg_cox_log_likelihood

from dl_toolbox.cross_validation.run_cv_cmd import run_cv_from_cmd
from dl_toolbox.cross_validation.cmd_line_parser import SurvivalCVCmdLineParser

from dl_toolbox.data.data_handler import DataHandler
from dl_toolbox.data.read import read_patient_data
from dl_toolbox.callbacks.concordance_index import ConcordanceIndex

from cv_train_from_scratch import FromScratchCVContext


# a custom data handler to read rgb images instead of grayscale
class RGBDataHandler(DataHandler):
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
                 preproc_fun=None):

        self.preproc_fun = preproc_fun
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
        super()._read_data()

        def rgb_preproc(ct_slices, roi, label):
            # convert from [0,1] range to [0, 255]
            rgb_range = np.zeros(ct_slices.shape, dtype="uint8")
            for i, ct in enumerate(ct_slices):
                rgb_range[i] = normalize_img(ct, new_low=0, new_up=255).astype("uint8")

            rgb = to_rgb(rgb_range)

            return rgb, roi, label

        # the preprocessing for the networks needs to be done!
        def network_preproc(ct_slices_rgb, roi, label):
            if self.preproc_fun is None:
                return ct_slices_rgb, roi, label

            ct_preproc = np.zeros(ct_slices_rgb.shape)
            for i, ct in enumerate(ct_slices_rgb):
                ct_preproc[i] = self.preproc_fun(ct)

            return ct_preproc, roi, label

        # now do the RGB conversion
        for pat in self.img_dict:
            # n_slices x height x width x 1
            img = self.img_dict[pat]['img']
            roi = self.img_dict[pat]['mask']
            lab = self.outcome_dict[pat]

            img_rgb, _, _ = rgb_preproc(img, roi, lab)
            img_pre, _, _ = network_preproc(img_rgb, roi, lab)

            ########## DEBUG:
            # f, ax = plt.subplots(2, 5, figsize=(15, 10))
            # slice_idx = len(img) // 2
            # ax[0, 0].imshow(img[slice_idx].squeeze(), cmap="gray")
            # ax[0, 0].set_title("original")

            # ax[0, 1].imshow(img_rgb[slice_idx])
            # ax[0, 1].set_title("after RGB")

            # ax[0, 2].imshow(img_rgb[slice_idx, ..., 0], cmap="gray")
            # ax[0, 2].set_title("channel 0 after RGB")

            # ax[0, 3].imshow(img_rgb[slice_idx, ..., 1], cmap="gray")
            # ax[0, 3].set_title("channel 1 after RGB")

            # ax[0, 4].imshow(img_rgb[slice_idx, ..., 2], cmap="gray")
            # ax[0, 4].set_title("channel 2 after RGB")


            # ax[1, 1].imshow(img_pre[slice_idx])
            # ax[1, 1].set_title("after RGB+preproc")
            # ax[1, 2].imshow(img_pre[slice_idx, ..., 0], cmap="gray")
            # ax[1, 2].set_title("channel 0 after RGB+preproc")
            # ax[1, 3].imshow(img_pre[slice_idx, ..., 1], cmap="gray")
            # ax[1, 3].set_title("channel 1 after RGB+preproc")
            # ax[1, 4].imshow(img_pre[slice_idx, ..., 2], cmap="gray")
            # ax[1, 4].set_title("channel 2 after RGB+preproc")
            # plt.show()
            ###########

            self.img_dict[pat]['img'] = img_pre
            # self.img_dict[pat]['mask'] = roi
            # self.outcome_dict[pat] = lab


class TransferLearningParser(SurvivalCVCmdLineParser):

    def add_context_args(self):
        super().add_context_args()

        architecture_group = self._get_or_create_group("Transfer")
        architecture_group.add_argument(
            "--architecture", type=str, help="Name of the pretrained model to use",
            choices=["VGG16", "ResNet50", "InceptionV3", "InceptionResNetV2",
                    "DenseNet121", "DenseNet169", "DenseNet201"],
            default="ResNet50")

        architecture_group.add_argument(
            "--layer", type=str, help="Name of the last pretrained layer to use.",
            default="")

    def get_context_args(self):
        args = super().get_context_args()

        architecture = self.parsed_args.architecture

        if architecture == "VGG16":
            arch_cls = vgg16.VGG16
            preproc_fun = getattr(vgg16, "preprocess_input")
        elif architecture == "ResNet50":
            arch_cls = resnet50.ResNet50
            preproc_fun = getattr(resnet50, "preprocess_input")
        elif architecture == "InceptionV3":
            arch_cls = inception_v3.InceptionV3
            preproc_fun = getattr(inception_v3, "preprocess_input")
        elif architecture == "InceptionResNetV2":
            arch_cls = inception_resnet_v2.InceptionResNetV2
            preproc_fun = getattr(inception_resnet_v2, "preprocess_input")
        elif architecture == "DenseNet121":
            arch_cls = densenet.DenseNet121
            preproc_fun = getattr(densenet, "preprocess_input")
        elif architecture == "DenseNet169":
            arch_cls = densenet.DenseNet169
            preproc_fun = getattr(densenet, "preprocess_input")
        elif architecture == "DenseNet201":
            arch_cls = densenet.DenseNet201
            preproc_fun = getattr(densenet, "preprocess_input")

        args["base_model_cls"] = arch_cls
        args["preproc_fun"] = preproc_fun

        layer = self.parsed_args.layer
        if layer == "":
            # this will be interpreted as the last layer before the dense
            # layers
            layer = None
        args["base_layer"] = layer

        return args

    def get_data_handler_args(self):
        """the preprocessing function is part of the DataHandler, not the cv context!"""
        args = super().get_data_handler_args()

        architecture = self.parsed_args.architecture

        if architecture == "VGG16":
            preproc_fun = getattr(vgg16, "preprocess_input")
        elif architecture == "ResNet50":
            preproc_fun = getattr(resnet50, "preprocess_input")
        elif architecture == "InceptionV3":
            preproc_fun = getattr(inception_v3, "preprocess_input")
        elif architecture == "InceptionResNetV2":
            preproc_fun = getattr(inception_resnet_v2, "preprocess_input")
        elif architecture == "DenseNet121":
            preproc_fun = getattr(densenet, "preprocess_input")
        elif architecture == "DenseNet169":
            preproc_fun = getattr(densenet, "preprocess_input")
        elif architecture == "DenseNet201":
            preproc_fun = getattr(densenet, "preprocess_input")

        args["preproc_fun"] = preproc_fun

        return args


# customize the cv context class for working with pretrained networks
class TransferLearningCVContext(FromScratchCVContext):
    def __init__(self,
                 data_handler,
                 seed=1,
                 train_ids=None,
                 train_fraction=None,
                 epochs=[10],
                 optimizer_cls=optimizers.Adam,
                 lr=[1.e-3],
                 loss=neg_cox_log_likelihood,
                 #### up to here base class args
                 batchnorm=None,
                 lrelu=0.,
                 l1=0.,
                 l2=0.,
                 dropout=0.,
                 finalact="tanh",
                 ##### up to here from_scratch_args
                 base_model_cls=densenet.DenseNet201,
                 base_layer=None,
                 base_pooling="avg",
                 dense_sizes=[128, 32, 1],
                 **kwargs):

        self.base_model_cls = base_model_cls
        self.base_layer = base_layer
        self.base_pooling = base_pooling
        self.dense_sizes = dense_sizes

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
        m = pretrained_model(
                input_shape=input_shape,
                base_model_cls=self.base_model_cls,
                last_base_layer_name=self.base_layer,
                freeze_all_pretrained=True,
                pooling=self.base_pooling,
                dense_sizes=self.dense_sizes,
                final_act=self.finalact,
                reg=regularizers.l1_l2(self.l1, self.l2),
                bn=self.batchnorm,
                lrelu=self.lrelu,
                name_prefix="added")

        m.compile(
            loss=self.loss,
            optimizer=self.optimizer_cls(lr=self.lr[0]))

        return m

    def train_function(self, compiled_model, train_generator,
                       valid_generator, callbacks):

        early_stop = keras_cb.EarlyStopping(
            monitor="val_loss", patience=5, verbose=1, mode="min")
        reduce_lr = keras_cb.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1, mode="min",
            min_lr=1e-7)
        cb_ci = ConcordanceIndex(
            (train_generator.x, train_generator.y),
            (valid_generator.x, valid_generator.y),
            freq=1)

        callbacks = [early_stop, reduce_lr, cb_ci] + callbacks

        # train in two phases

        # first train only the dense layers (the top of the model)
        # print("Training only newly added layers for {} epochs (lr={})!".format(
        #     self.parser.arg_dict["training"]["epochs"][0],
        #     K.eval(compiled_model.optimizer.lr)))

        # hist_top = compiled_model.fit_generator(
        #     train_generator,
        #     steps_per_epoch=np.ceil(
        #         train_generator.n / train_generator.batch_size),
        #     epochs=self.parser.arg_dict["training"]["epochs"][0],
        #     verbose=1,
        #     validation_data=valid_generator,
        #     validation_steps=np.ceil(
        #         valid_generator.n / valid_generator.batch_size),
        #     callbacks=callbacks
        # )

        # secondly unfreeze all layers and train them for a longer while
        # with a lower learning rate
        print("Training all layers for {} epochs (lr={})!".format(
            self.epochs[1], self.lr[1]))
        # now unfreeze all layers
        # this needs recompilation of the model though
        for layer in compiled_model.layers:
            layer.trainable = True

        # we want to train again but with a different learning rate
        # and for a different number of epochs
        new_opti = self.optimizer_cls(lr=self.lr[1])
        compiled_model.compile(
            optimizer=new_opti,
            loss=self.loss)

        hist = compiled_model.fit_generator(
            train_generator,
            steps_per_epoch=np.ceil(
                train_generator.n / train_generator.batch_size),
            epochs=self.epochs[1],
            verbose=1,
            validation_data=valid_generator,
            validation_steps=np.ceil(
                valid_generator.n / valid_generator.batch_size),
            callbacks=callbacks
        )

        # return [hist_top, hist]
        return [hist]


if __name__ == "__main__":
    run_cv_from_cmd(
        cv_context_cls=TransferLearningCVContext,
        data_handler_cls=RGBDataHandler,
        parser_cls=TransferLearningParser)
