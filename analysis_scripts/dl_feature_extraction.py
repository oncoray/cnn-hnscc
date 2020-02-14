import argparse
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

from keras import Model
from keras import backend as K
from keras.applications import resnet50, densenet, inception_resnet_v2

from dl_toolbox.models.transfer_learning import get_base_model
from dl_toolbox.data.read import read_patient_data
from dl_toolbox.preprocessing.stratification import exclude_duplicate_dktk_ids
from dl_toolbox.preprocessing import stack_slices_for_patients, to_rgb, normalize_img

from dl_toolbox.cross_validation.cmd_line_parser import SurvivalCVCmdLineParser


def extract_features_flattened(model, data):
    features = model.predict(data, verbose=1)
    # save original shape before flattening (first dimension is always the samples)
    original_shape = features.shape
    print("original_shape", original_shape)

    # now flatten the features and append them to the df
    features_flat = np.reshape(features, (original_shape[0], -1))
    print("features_flat shape", features_flat.shape)
    df_features = pd.DataFrame(features_flat)
    df_features.columns = ["feat_"+str(i+1) for i in range(features_flat.shape[1])]

    return df_features, original_shape


def extract_feature_info(labels, slice_descriptions, train_ids,
                         id_col, time_col, event_col):
    """
    Create a dataframe that contains patient ids, their outcome for every
    slice of that patient.

    Parameters
    ----------
    labels: np.array(2D) with time, event as columns
    slice_descriptions: list of str
        each string is assumed to be of the form <patient_id>_slice_<slice_nr>
    train_ids: list
        list of patient ids which will be assigned a 'cohort' column entry
        of 'training', all others will be assigned the value 'test'
    """
    return pd.DataFrame({
        id_col: [d.split("_slice_")[0] for d in slice_descriptions],
        'slice_idx': [d.split("_slice_")[1] for d in slice_descriptions],
        time_col: [l[0] for l in labels],
        event_col: [l[1] for l in labels],
        'cohort': ["training" if d.split("_slice_")[0] in train_ids else "test"
                   for d in slice_descriptions]
    })


def extract_features_from_pretrained_model(data, labels, slice_descriptions,
                                           train_ids, pretrained_model_str,
                                           layer_name=None, drop_small_variance=True):

    if pretrained_model_str.lower().startswith("resnet"):
        model_fun = getattr(resnet50, pretrained_model_str)
        preproc_fun = getattr(resnet50, "preprocess_input")
    elif pretrained_model_str.lower().startswith("densenet"):
        model_fun = getattr(densenet, pretrained_model_str)
        preproc_fun = getattr(densenet, "preprocess_input")
    elif pretrained_model_str.lower().startswith("inceptionresnetv2"):
        model_fun = getattr(inception_resnet_v2, pretrained_model_str)
        preproc_fun = getattr(inception_resnet_v2, "preprocess_input")
    else:
        raise ValueError(
            "Unknown model string {}: Choose a name contained in keras.applications.resnet50'"
            " 'keras.applications.densenet' or 'keras.applications.inception_resnet_v2'!".format(pretrained_model_str))

    K.clear_session()

    model = get_base_model(
        input_shape=data.shape[1:],
        base_model_cls=model_fun,
        layer_name=layer_name,
        freeze_all_pretrained=True,
    )
    print(model.summary(line_length=120))

    # use the preprocessing function provided by keras
    X = np.zeros(data.shape)
    for i in range(len(X)):
        # print("preproc_fun called with data_shape {} and range {}-{}, dtype={}".format(data[i].shape, data[i].min(), data[i].max(), data[i].dtype))
        X[i] = preproc_fun(data[i])

    df_features, original_shape = extract_features_flattened(model, X)
    n_feats = df_features.shape[1]

    # compute variances only on the training data (otherwise information is leaked)
    df_info = extract_feature_info(labels, slice_descriptions, train_ids)
    df_features_train = df_features[df_info.id.isin(train_ids)]
    # compute how many of the features are constant or low variance
    train_features = df_features_train.values
    variances = np.var(train_features, axis=0)
    print("min(variances)={}, max(variances)={}, mean(variances)={} computed on training set.".format(
        np.min(variances), np.max(variances), np.mean(variances)))

    if drop_small_variance:
        var_thresh = 1e-1
        usable_cols = np.where(variances > var_thresh)[0]
        print("{}/{} of features have variance <= {} in training set ({}%) and were dropped!".format(
            n_feats - len(usable_cols), n_feats, var_thresh,
            np.round(100 * (n_feats - len(usable_cols)) / n_feats, 2)))

    else:
        # all feature columns are usable
        usable_cols = list(range(n_feats))

    df_features = df_features.iloc[:, usable_cols]
    print(df_info.shape, df_features.shape)
    df = pd.concat([df_info, df_features], axis=1)
    print(df.shape)
    print(df.head(2))

    statistics = pd.DataFrame({
        'model': [pretrained_model_str],
        'layer': [layer_name] if layer_name is not None else ["last"],
        'variance_thresh': [var_thresh],
        'feature_variance_min': [np.min(variances)],
        'feature_variance_median': [np.median(variances)],
        'feature_variance_mean': [np.mean(variances)],
        'feature_variance_max': [np.max(variances)],
        'feature_shape': [original_shape[1:]],
        'total_features': [n_feats],
        'features_above_thresh': [df_features.shape[1]],
        'used_percentage': [np.round(100 * df_features.shape[1] / n_feats, 2)]
    })

    return df, feature_shape[1:], statistics
