import numpy as np
import os

from lifelines.statistics import logrank_test
from functools import partial, update_wrapper


def check_cohort_differences(ids_A, ids_B, outcomes):
    # are sets A and B significantly different regarding KM curves?
    test_res = logrank_test(
        np.array([outcomes[pat][0] for pat in ids_A]),
        np.array([outcomes[pat][0] for pat in ids_B]),
        event_observed_A=np.array([outcomes[pat][1] for pat in ids_A]),
        event_observed_B=np.array([outcomes[pat][1] for pat in ids_B]))

    print("P-value for significant differences in Kaplan Meier curves for both sets: {}".format(
        test_res.p_value))

    return test_res


def parse_list_from_string(s, dtype=float, sep=", "):
    """
    When reading back pandas Dataframes from csv files
    and having a column that stores an array, they are read as strings
    (e.g. '[12, 14, 59]')
    to get it back to a list one has to do this parsing
    """
    elem_str = s.split("[")[1].split("]")[0]
    elem_split = elem_str.split(sep)
    # print("elem_split={}".format(elem_split))
    return [dtype(elem.strip()) for elem in elem_split if elem] # discard possibly empty strings left over from the split


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)

    return partial_func


def subdirectories_full_path(path):
    return [os.path.join(path, d) for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))]


def subdirectories_full_path_condition(path, condition):
    """
    condition based on full path
    """
    subdirs = subdirectories_full_path(path)
    return [d for d in subdirs if condition(d)]


def repetition_dirs(path):
    return subdirectories_full_path_condition(
        path, condition=lambda d: os.path.basename(d).startswith("rep_"))


def fold_dirs(path):
    return subdirectories_full_path_condition(
        path, condition=lambda d: os.path.basename(d).startswith("fold_"))


def get_model_paths(model_cv_dir):
    model_paths = []
    for rep_path in repetition_dirs(model_cv_dir):
        for fold_path in fold_dirs(rep_path):
            model_paths.append(
                os.path.join(fold_path, "trained_model.h5"))
    return model_paths
