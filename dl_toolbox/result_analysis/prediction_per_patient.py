from ..preprocessing.preprocessing import to_rgb

import numpy as np
import pandas as pd


def model_predictions_per_patient(model, data_dict, to_rgb=False,
                                  avg_method="mean", output_idx=0,
                                  additional_feature_df=None,
                                  id_col="id"):
    """
    Use the model to make a single prediction per patient by averaging the results over the slices.

    Parameters
    ----------
    data_dict needs to be a dictionary with keys = patient_ids and values = dictionary with keys 'ct', 'roi'
    that both contain 4D numpy arrays of equal shape that is N_slices x H x W x C.
    The last three dimensions should be the same across all patients whereas the number of slices
    can differ.

    avg_method: either "mean" or "median" or a callable that receives a 1D np.array and returns a single number
        The method used for aggregating the slice results to a single patient result

    output_idx: int
        in case the model has multiple outputs, which one should be used

    additional_feature_df: None or pd.DataFrame
        in case the model expects additional input, this can be provided here
    id_col: str
        patient id column name in additonal_feature_df and in the output dataframe

    Returns
    -------
    A pd.DataFrame with an entry for every patient id with the following columns:
        'pred_per_slice': the prediction for each slice of the patient
        'pred_per_pat(<avg_method>)': the aggregate of the predictions over the individual slices
        'pred_variance': the variance of the predictions over the individual slices
    """

    if avg_method == "mean":
        avg_fn = np.mean
    elif avg_method == "median":
        avg_fn = np.median
    else:
        avg_fn = avg_method
    # else:
    #     raise ValueError("Unknown avg_method passed. Use 'mean' or 'median'!")

    ids = list(data_dict.keys())
    slice_idx = [None] * len(ids)
    pred_per_slice = [None] * len(ids)
    for i, pat_id in enumerate(ids):
        imgs = data_dict[pat_id]['img']
        # convert to RGB if the imgs don't have 3 color channels (assumes channels_last)
        if to_rgb and imgs.shape[-1] != 3:
            imgs_rgb = to_rgb(imgs)
        else:
            imgs_rgb = imgs

        model_input = [imgs_rgb]
        if additional_feature_df is not None:
            pat_features = additional_feature_df[
                additional_feature_df[id_col] == pat_id].drop(
                    id_col, axis=1).values[0]

            # need replication to the number of slices
            pat_features = np.tile(pat_features, (imgs_rgb.shape[0], 1))
            model_input.append(pat_features)

        # predictions for each slice
        # is a 2d array of n_images x 1 thats why we discard second dimension
        pred = model.predict(model_input)
        if isinstance(pred, list):
            pred = pred[output_idx]
        pred = pred.flatten()

        # store results
        slice_idx[i] = data_dict[pat_id]['slice_idx']
        pred_per_slice[i] = pred

    return pd.DataFrame({
        id_col: ids,
        'slice_idx': slice_idx,
        'pred_per_slice': pred_per_slice,
        'pred_per_pat({})'.format(avg_method): [
            avg_fn(slice_preds) for slice_preds in pred_per_slice],
        'pred_variance': [
            np.var(slice_preds) for slice_preds in pred_per_slice]
        })
