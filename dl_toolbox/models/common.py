from keras.layers import Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose, BatchNormalization, Activation, Dense, LeakyReLU


def dense_bn_act(input_tensor, n_dense, activation="relu", bn=None, dense_args={}, name=""):
    """
    activation: either string or an instance of a keras activation that can be called
                on a tensor
    dense_args: dictionary containing keys and values for e.g. kernel_initializer, bias_initializer, ...
    """
    act = Activation(activation) if isinstance(activation, str) else activation
    act.name = name + "_act"

    x = Dense(n_dense, name=name+"_dense", **dense_args)(input_tensor)
    if bn is None:
        x = act(x)
    elif bn == "pre_act":
        x = BatchNormalization(name=name+"_bn")(x)
        x = act(x)
    elif bn == "post_act":
        x = act(x)
        x = BatchNormalization(name=name+"_bn")(x)
    return x


def dense_lrelu(input_tensor, n_dense, lrelu=.0, bn=None, dense_args={}, name=""):
    return dense_bn_act(input_tensor, n_dense, activation=LeakyReLU(lrelu), bn=bn, dense_args=dense_args, name=name)


def conv_bn_act(input_tensor, conv_layer, n_filters, filter_size=3, strides=1, activation="relu", bn=None, conv_args={}, name=""):
    act = Activation(activation) if isinstance(activation, str) else activation
    act.name = name + "_act"

    x = conv_layer(n_filters, filter_size, name=name+"_conv", **conv_args, strides=strides)(input_tensor)
    if bn is None:
        x = act(x)
    elif bn == "pre_act":
        x = BatchNormalization(name=name+"_bn")(x)
        x = act(x)
    elif bn == "post_act":
        x = act(x)
        x = BatchNormalization(name=name+"_bn")(x)
    return x


def conv_lrelu_3d(input_tensor, n_filters, strides=1, filter_size=3, lrelu=0., bn=None, conv_args={}, name=""):
    return conv_bn_act(
        input_tensor,
        conv_layer=Conv3D, n_filters=n_filters, filter_size=filter_size, strides=strides,
        activation=LeakyReLU(lrelu), bn=bn, conv_args=conv_args, name=name)


def conv_transpose_lrelu_3d(input_tensor, n_filters, strides=1, filter_size=3, lrelu=0., bn=None, conv_args={}, name=""):
    return conv_bn_act(
        input_tensor,
        conv_layer=Conv3DTranspose, n_filters=n_filters, filter_size=filter_size, strides=strides,
        activation=LeakyReLU(lrelu), bn=bn, conv_args=conv_args, name=name)


def conv_lrelu_2d(input_tensor, n_filters, strides=1, filter_size=3, lrelu=0., bn=None, conv_args={}, name=""):
    return conv_bn_act(
        input_tensor,
        conv_layer=Conv2D, n_filters=n_filters, filter_size=filter_size, strides=strides,
        activation=LeakyReLU(lrelu), bn=bn, conv_args=conv_args, name=name)


def conv_transpose_lrelu_2d(input_tensor, n_filters, strides=1, filter_size=3, lrelu=0., bn=None, conv_args={}, name=""):
    return conv_bn_act(
        input_tensor,
        conv_layer=Conv2DTranspose, n_filters=n_filters, filter_size=filter_size, strides=strides,
        activation=LeakyReLU(lrelu), bn=bn, conv_args=conv_args, name=name)
