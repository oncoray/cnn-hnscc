import matplotlib
matplotlib.use("Agg")   # non interactive backend for not stopping computation for plotting

import numpy as np

from keras import layers, Model, regularizers, optimizers
from keras import callbacks as keras_cb

from dl_toolbox.cross_validation import SurvivalCVContext, run_cv_from_cmd

from dl_toolbox.models.common import dense_bn_act, dense_lrelu, conv_lrelu_2d

from dl_toolbox.callbacks.concordance_index import ConcordanceIndex
from dl_toolbox.losses import neg_cox_log_likelihood


class FromScratchCVContext(SurvivalCVContext):
    def __init__(self,
                 data_handler,
                 seed=1,
                 train_ids=None,
                 train_fraction=None,
                 epochs=[10],
                 optimizer_cls=optimizers.Adam,
                 lr=[1.e-3],
                 loss=neg_cox_log_likelihood,
                 batchnorm=None,
                 lrelu=0.,
                 l1=0.,
                 l2=0.,
                 dropout=0.,
                 finalact="tanh",
                 **kwargs):

        self.batchnorm = batchnorm
        self.lrelu = lrelu
        self.l1 = l1
        self.l2 = l2
        self.dropout = dropout
        self.finalact = finalact

        super().__init__(
            data_handler,
            seed=seed,
            train_ids=train_ids,
            train_fraction=train_fraction,
            epochs=epochs,
            optimizer_cls=optimizer_cls,
            lr=lr,
            loss=loss,
            **kwargs)

    def create_compiled_model(self, input_shape):

        bn = self.batchnorm
        lrelu = self.lrelu

        reg = regularizers.l1_l2(self.l1, self.l2)

        dr = self.dropout
        final_act = self.finalact

        args = {'kernel_regularizer': reg}
        conv_args = {'kernel_regularizer': reg, 'padding': 'same'}

        inputs = layers.Input(input_shape)
        n_filters = 16

        x = inputs
        for i in range(5):
            filter_size = (5, 5) if i == 0 else (3, 3)

            x = conv_lrelu_2d(
                x, n_filters, strides=1, filter_size=filter_size,
                lrelu=lrelu, bn=bn, conv_args=conv_args, name="conv_{}".format(
                    2*i+1))

            x = conv_lrelu_2d(
                x, n_filters, strides=2, filter_size=filter_size,
                lrelu=lrelu, bn=bn, conv_args=conv_args, name="conv_{}".format(
                    2*i+2))

            n_filters *= 2

        # this seems to work better than GlobalMaxPooling
        x = layers.Flatten()(x)
        x = layers.Dropout(dr)(x)
        x = dense_lrelu(x, 256, lrelu=lrelu, bn=bn, dense_args=args, name="dense_1")
        x = layers.Dropout(dr)(x)
        x = dense_lrelu(x, 64, lrelu=lrelu, bn=bn, dense_args=args, name="dense_2")
        x = dense_bn_act(x, 1, activation=final_act, bn=bn, dense_args=args,
                         name="cox_out")

        m = Model(inputs, x)

        m.compile(
            loss=self.loss,
            optimizer=self.optimizer_cls(lr=self.lr[0]))

        return m

    def train_function(self, compiled_model, train_generator,
                       valid_generator, callbacks):

        early_stop = keras_cb.EarlyStopping(
            monitor="val_loss", patience=10, verbose=1, mode="min")
        reduce_lr = keras_cb.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1, mode="min",
            min_lr=1e-7)
        cb_ci = ConcordanceIndex(
            (train_generator.x, train_generator.y),
            (valid_generator.x, valid_generator.y),
            freq=1)

        callbacks = [
            early_stop, reduce_lr,
            cb_ci] + callbacks

        return super().train_function(
            compiled_model, train_generator, valid_generator, callbacks)


if __name__ == "__main__":
    run_cv_from_cmd(
        cv_context_cls=FromScratchCVContext,
        ensemble_method=np.mean)
