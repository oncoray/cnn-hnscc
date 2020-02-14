import itertools
import nibabel as nib
import numpy as np
import os
import pandas as pd

from ..preprocessing import pandas_outcome_to_dict,\
    extract_slices_around_largest_tumor
from ..preprocessing.stratification import exclude_duplicate_dktk_ids


def read_nifty_as_np(file):
    img = nib.load(file)
    return img.get_fdata()


def get_roi_and_ct_array(src_dir, pat_id):
    # for a single patient, set filepaths to nifty files
    ct_dir = os.path.join(src_dir, pat_id, 'CTCT_GTV_Plan/')

    ct_filename = next(os.walk(ct_dir))[2][0]
    ct_file = os.path.join(ct_dir, ct_filename)
    # print(ct_file)
    ct = read_nifty_as_np(ct_file)

    roi_filename = next(os.walk(ct_dir))[2][1]
    roi_file = os.path.join(ct_dir, roi_filename)
    # print(roi_file)
    roi = read_nifty_as_np(roi_file)

    assert roi.shape == ct.shape

    return ct, roi


def read_outcome(outcome_file, id_col, time_col, event_col, dropna=True, csv_sep=";"):
    outcome = pd.read_csv(outcome_file, sep=csv_sep)
    print("\noutcome shape: {}".format(outcome.shape))

    if dropna:
        len_old = len(outcome)
        outcome = outcome[[id_col, time_col, event_col]].dropna()
        len_new = len(outcome)
        if len_old != len_new:
            print("Dropped {} patients due to missing outcome!".format(
                len_old - len_new))

    outcome_dict = pandas_outcome_to_dict(
            outcome,
            id_col=id_col,
            survival_col=time_col,
            event_col=event_col)

    # provide some statistics on distribution of times for each cohort
    all_times = [outcome_dict[pat][0] for pat in outcome_dict]
    event_times = [
        outcome_dict[pat][0] for pat in outcome_dict
        if outcome_dict[pat][1] == 1]
    censor_times = [
        outcome_dict[pat][0] for pat in outcome_dict
        if outcome_dict[pat][1] == 0]

    print("{} patients with outcome, {} with events ({}%)".format(
        len(all_times), len(event_times),
        np.round(100 * len(event_times) / len(all_times), 2)))

    tpl = "{} distribution: min = {}, median = {}, mean = {}, max = {}"
    for part, time in zip(["all_times", "event_times", "censor_times"], [all_times, event_times, censor_times]):
        print(tpl.format(
            part, np.min(time), np.median(time), np.mean(time), np.max(time)))

    return outcome_dict


def read_img_data(img_directories, n_around_max, whitelist, img_fn="ct", mask_fn="roi"):
    """
    Parameters
    ----------
    img_directories: list
        directories that contain image data for each patient in separate folder
        as numpy files (either .npy or .npz) with uniform filenames given by
        below arguments.
        If both npy and npz files are present, the npz files are loaded due to increased
        speed.
    img_fn: str
        name of the files that contain the image data for each patient (without file extension)
    mask_fn: str
        name of the files that contain the mask data for each patient (without file extension).
    """
    # read images
    img_data = {}
    for img_dir in img_directories:
        pat_dirs = sorted([
            os.path.join(img_dir, d) for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))])

        count = 0
        for pat_path in pat_dirs:
            pat = os.path.basename(pat_path)
            if pat not in whitelist:
                print("\nskipping", pat, "because no outcome available")
                continue

            #time, event = outcomes[cohort][pat]
            #if event == 0 and time < 12:
            #    print("skipping", pat, "because censored before 12 months: time={}, event={}".format(
            #        time, event))
            #    continue

            # determine if we have npy or npz files and load accordingly
            # if both are there, prefer npz
            file_extensions_ct = [
                os.path.splitext(f)[1] for f in os.listdir(pat_path) if f.startswith(img_fn)]
            file_extensions_roi = [
                os.path.splitext(f)[1] for f in os.listdir(pat_path) if f.startswith(mask_fn)]

            if ".npz" in file_extensions_ct:
                ct = np.load(os.path.join(pat_path, img_fn + ".npz"))["arr_0"]
            else:
                ct = np.load(os.path.join(pat_path, img_fn + ".npy"))
            #print("read CT for", pat)
            if ".npz" in file_extensions_roi:
                roi = np.load(os.path.join(pat_path, mask_fn + ".npz"))["arr_0"]
            else:
                roi = np.load(os.path.join(pat_path, mask_fn + ".npy"))
            #print("read mask for", pat)

            # Now only select the necessary slices
            ct, roi, slice_idx = extract_slices_around_largest_tumor(
                ct, roi, n_around_max)

            img_data[pat] = {
                'img': ct, 'mask': roi, 'slice_idx': slice_idx}

            print("\r{}/{}: {} {}, {}\t".format(
                count+1, len(pat_dirs), pat, ct.shape, roi.shape),
                  end="", flush=True)

            count += 1

        print("\n{}: read images of {} patients".format(
            img_dir, count))

    print("Total number of image data", len(list(img_data.keys())))

    return img_data


def read_baseline_feats(baseline_feat_files, whitelist=None, id_col="id",
                        drop_cols=None):
    """
    baseline_feat_files: list of str
        paths to csv files

    drop_cols: list of str
        list of column names that should not be considered as features and be dropped
        (id_col is always ignored and does not need to be passed!)

    """
    if not isinstance(baseline_feat_files, list):
        baseline_feat_files = [baseline_feat_files]

    baseline_feats = [None] * len(baseline_feat_files)

    for i, feat_file in enumerate(baseline_feat_files):
        baseline_df = pd.read_csv(feat_file)
        # all columns except id column are taken as features

        # use only those with outcome
        if whitelist is not None:
            df_whitelist = baseline_df[baseline_df[id_col].isin(whitelist)]
        else:
            df_whitelist = baseline_df

        if drop_cols is not None:
            df_whitelist.drop(drop_cols, axis=1, inplace=True)

        baseline_feats[i] = df_whitelist

    df = pd.concat(baseline_feats, axis=0, ignore_index=True)

    print("read baseline_feats of {} patients".format(len(df)))

    return df


def read_outcome_and_baseline_features(outcome_file,
                                       time_col,
                                       event_col,
                                       id_col,
                                       csv_sep=";",
                                       baseline_feat_filename=None):

    outcomes = read_outcome(
        outcome_file, id_col, time_col, event_col, dropna=True, csv_sep=csv_sep)

    all_ids = list(outcomes.keys())

    # we have to remove some patients of the DKTK
    all_ids = exclude_duplicate_dktk_ids(all_ids)

    # baseline_feats
    if baseline_feat_filename is not None:
        baseline_feats = read_baseline_feats(
            baseline_feat_filename, whitelist=all_ids,
            id_col=id_col)
    else:
        baseline_feats = None

    return outcomes, baseline_feats


def read_patient_data(img_directories,
                      outcome_file,
                      time_col,
                      event_col,
                      id_col,
                      n_around_max,
                      preproc_fun=None,
                      csv_sep=";"):
    """
    A wrapper to quickly read all relevant data which is
    used in our lab.
    """
    outcomes = read_outcome(
        outcome_file, id_col, time_col, event_col, dropna=True, csv_sep=csv_sep)

    all_ids = list(outcomes.keys())
    img_data = read_img_data(
        img_directories, n_around_max, whitelist=all_ids)

    # check that we have outcome available for all image data
    assert set(list(img_data.keys())).issubset(set(all_ids))

    # preprocessing if a function was given
    if preproc_fun is not None:
        for pat in img_data:
            # n_slices x height x width x 1
            img = img_data[pat]['img']
            roi = img_data[pat]['mask']
            lab = outcomes[pat]

            img_new, roi_new, lab_new = preproc_fun(img, roi, lab)
            img_data[pat]['img'] = img_new
            img_data[pat]['mask'] = roi_new
            outcomes[pat] = lab_new


    return outcomes, img_data
