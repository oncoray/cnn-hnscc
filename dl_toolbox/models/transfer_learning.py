from .common import dense_lrelu, dense_bn_act

from keras import Model
from keras.layers import Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D


def get_model_part(base_model, layer_name=None):
    """
    Return a part of a given model up to a certain layer name.
    """
    if layer_name is None:
        base_out = base_model.get_output_at(0)
    else:
        last_layer = base_model.get_layer(layer_name)
        base_out = last_layer.get_output_at(0)

    return Model(inputs=base_model.inputs, outputs=base_out)


def get_base_model(input_shape, base_model_cls, layer_name=None, freeze_all_pretrained=True):
    """
    Returns a keras.applications model up to a given layer name.
    """
    base_model = base_model_cls(
        input_shape=input_shape, weights='imagenet', include_top=False,
        pooling=None)

    if freeze_all_pretrained:
        for layer in base_model.layers:
            layer.trainable = False


    return get_model_part(base_model, layer_name)


def pretrained_model(input_shape, base_model_cls,
                     last_base_layer_name,
                     freeze_all_pretrained=True,
                     pooling=None,
                     dense_sizes=[128, 1],
                     final_act=None,
                     bn=None,
                     reg=None,
                     lrelu=0.,
                     name_prefix=""):

    """
    Take a pretrained imagenet model and modify only the fully connected
    block and train this.

    Parameters
    ----------
    input_shape: np.array
    base_model_cls: one of the functions used for creating models
        should be one of the keras.applications classes (VGG16, ResNet50, ...)
    last_base_layer_name: str or None
        needs to be a name of a layer of the base_model.
        Up to this layer, the base_model will not be trained.
        If None, the last layer of the model before the dense layers
        will be used.

    """

    base = get_base_model(
        input_shape=input_shape, base_model_cls=base_model_cls,
        layer_name=last_base_layer_name,
        freeze_all_pretrained=freeze_all_pretrained)

    x = base.get_output_at(0)

    if pooling is None:
        x = Flatten(name=name_prefix + "_flatten")(x)
    elif pooling == "avg":
        x = GlobalAveragePooling2D(name=name_prefix + "_avg")(x)
    elif pooling == "max":
        x = GlobalMaxPooling2D(name_prefix + "_max")(x)

    dense_args = dict(kernel_regularizer=reg)
    for i, size in enumerate(dense_sizes[:-1]):
        x = dense_lrelu(
            x, size, bn=bn, lrelu=lrelu, dense_args=dense_args,
            name=name_prefix + "_finetune_dense_" + str(i))

    x = dense_bn_act(
        x, dense_sizes[-1], activation=final_act,
        dense_args=dense_args,
        name=name_prefix + "_finetune_dense_out")

    model = Model(inputs=base.inputs, outputs=x)

    return model
