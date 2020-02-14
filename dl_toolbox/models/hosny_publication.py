from keras import Model, layers, regularizers
from .common import conv_lrelu_3d, conv_lrelu_2d, dense_bn_act, dense_lrelu

import numpy as np


def _hosny_model(version_str, input_shape=(50, 50, 50, 1), n_outputs=2,
                 final_act="softmax", dropout=[0.25, 0.5],
                 lrelu=0.1, bn="pre_act", l1_reg=0., l2_reg=1.e-5,
                 pool_layer_after_conv=layers.Flatten):
    """
    version_str: str, either "2D" or "3D"
    """
    assert version_str in ["2D", "3D"]
    if version_str == "2D":
        conv_lrelu_cls = conv_lrelu_2d
        pool_cls = layers.MaxPooling2D
    else:
        conv_lrelu_cls = conv_lrelu_3d
        pool_cls = layers.MaxPooling3D

    regularizer = regularizers.l1_l2(l1_reg, l2_reg)
    conv_args = {
        'kernel_regularizer': regularizer,
        'padding': 'valid'}
    dense_args = {
        'kernel_regularizer': regularizer}

    dr_conv = dropout[0]
    dr_fc = dropout[1]

    input = layers.Input(input_shape)
    x = conv_lrelu_cls(input, n_filters=64, filter_size=5, lrelu=lrelu,
                       bn=bn, conv_args=conv_args, name="conv1")
    x = layers.Dropout(dr_conv)(x)

    x = conv_lrelu_cls(x, n_filters=128, filter_size=3, lrelu=lrelu,
                       bn=bn, conv_args=conv_args, name="conv2")

    x = pool_cls(pool_size=3, name="pool1")(x)
    x = layers.Dropout(dr_conv)(x)

    x = conv_lrelu_cls(x, n_filters=256, filter_size=3, lrelu=lrelu,
                       bn=bn, conv_args=conv_args, name="conv3")
    x = layers.Dropout(dr_conv)(x)

    x = conv_lrelu_cls(x, n_filters=512, filter_size=3, lrelu=lrelu,
                       bn=bn, conv_args=conv_args, name="conv4")
    x = pool_cls(pool_size=3, name="pool2")(x)
    x = layers.Dropout(dr_conv)(x)

    x = pool_layer_after_conv(name="fc1")(x)

    x = dense_lrelu(x, 512, lrelu=lrelu, bn=bn,
                    dense_args=dense_args, name="fc2")
    x = layers.Dropout(dr_fc)(x)

    x = dense_lrelu(x, 256, lrelu=lrelu, bn=bn,
                    dense_args=dense_args, name="fc3")
    x = layers.Dropout(dr_fc)(x)
    x = dense_bn_act(x, n_outputs, activation=final_act, bn=bn,
                     dense_args=dense_args, name="fc4")

    return Model(input, x)


def hosny_model_3d(input_shape=(50, 50, 50, 1), n_outputs=2,
                   final_act="softmax", dropout=[0.25, 0.5],
                   lrelu=0.1, bn="pre_act", l1_reg=0., l2_reg=1.e-5,
                   weights=None, pool_layer_after_conv=layers.Flatten):
    """
    Parameters
    ----------
    weights: None or path to the model weight file
    """

    # the weights can be loaded from
    # https://github.com/modelhub-ai/deep-prognosis/blob/master/contrib_src/model/weights.h5"

    # according to their paper they used the following hyperparameters during training:
    #   optimizer = Adam(lr=1e-3)
    #   batch_size = 16
    # and for finetuning on the surgery cohort only the fc4 layer was trained again
    #   lr=1e-2
    #   batch_size = 24
    # augmentation factor of 32000 (random translations up to 10 pixels,
    # random 90 degree rotations around z axis, flipping along all axes)
    model = _hosny_model(version_str="3D", input_shape=input_shape,
                         n_outputs=n_outputs,
                         final_act=final_act, dropout=dropout, lrelu=lrelu,
                         bn=bn, l1_reg=l1_reg, l2_reg=l2_reg,
                         pool_layer_after_conv=pool_layer_after_conv)

    if weights is not None:
        if input_shape != (50, 50, 50, 1):
            print(
                "[W]: weights can only be loaded for input shape (50, 50, 50, 1)!"
                " Will not load weights.")
        else:
            try:
                model.load_weights(weights)
                print("Loaded model weights from {}".format(weights))
            except Exception as e:
                print("Loading weights from {} failed! {}".format(
                    weights, e))

    return model


def hosny_model_3d_with_restored_conv_weights(weights,
                                              input_shape=(50, 50, 50, 1),
                                              n_outputs=2, final_act="softmax",
                                              dropout=[0.25, 0.5], lrelu=0.1,
                                              bn="pre_act", l1_reg=0.,
                                              l2_reg=1.e-5,
                                              pool_layer_after_conv=layers.Flatten):
    """
    Uses the pretrained weights of the hosny publication for all
    convolutional layers, even though the input/output shapes might
    be different from the original publication.

    Parameters
    ----------
    weights: str
        path to the weights.h5 of the original publication
    """

    # load the original model with weights
    m_pretrained = hosny_model_3d(weights=weights)
    # then create the model we actually want
    m = hosny_model_3d(input_shape=input_shape, n_outputs=n_outputs,
                       final_act=final_act, dropout=dropout, lrelu=lrelu,
                       bn=bn, l1_reg=l1_reg, l2_reg=l2_reg,
                       pool_layer_after_conv=pool_layer_after_conv)

    # all conv layers should be transferable
    for i, layer in enumerate(m.layers):
        if layer.name.startswith("conv") and layer.count_params() > 0:
            pretrained_layer = m_pretrained.layers[i]
            assert layer.name == pretrained_layer.name

            pretrained_layer_weights = pretrained_layer.get_weights()
            layer.set_weights(pretrained_layer_weights)

            # weights and biases
            for j, tensor in enumerate(m.layers[i].get_weights()):
                assert np.all(
                    tensor == pretrained_layer_weights[j])
            print(f"Transfered weights for layer {i}:{layer.name}")

    return m


def hosny_model_2d(input_shape=(224, 224, 1), n_outputs=2,
                   final_act="softmax", dropout=[0.25, 0.5],
                   lrelu=0.1, bn="pre_act", l1_reg=0., l2_reg=1.e-5,
                   pool_layer_after_conv=layers.Flatten):

    # NOTE: due to possibly much larger input shapes in 2D
    # the model becomes huge since flattening is used instead
    # of global pooling!
    return _hosny_model(version_str="2D", input_shape=input_shape,
                        n_outputs=n_outputs,
                        final_act=final_act, dropout=dropout, lrelu=lrelu,
                        bn=bn, l1_reg=l1_reg, l2_reg=l2_reg,
                        pool_layer_after_conv=pool_layer_after_conv)
