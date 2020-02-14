import numpy as np
from itertools import chain
from sklearn.model_selection import RepeatedStratifiedKFold


def exclude_duplicate_dktk_ids(dktk_ids):
    # after inspection we found out that some IDs are actually the same
    # patient so we have to exclude them from the valid ids
    training_excludes = ["DKTK42"]
    valid_excludes = [
        "FDG02", "FDG08", "FDG11", "FDG12", "FDG25", "FDG27", "FDG30", "FDG31", "FDG32", "FDG41"]

    excludes = training_excludes + valid_excludes

    reduced = [id for id in dktk_ids if id not in excludes]

    if len(reduced) < len(dktk_ids):
        print("WARNING: excluded {} ids from dktk_ids since they refer to the same patient as other ids".format(
            len(dktk_ids) - len(reduced)))

    return reduced

def stratify_dktk_cohort(dktk_ids):

    dktk_ids = exclude_duplicate_dktk_ids(dktk_ids)

    train_ids = [pat_id for pat_id in dktk_ids if pat_id.startswith("DKTK")]
    print("{}/{} patients for training".format(len(train_ids), len(dktk_ids)))

    valid_ids = list(set(dktk_ids) - set(train_ids))
    print("{}/{} patients for validation".format(len(valid_ids), len(dktk_ids)))

    return train_ids, valid_ids


def stratify_dktk_three_way(dktk_ids, valid_fraction=0.2, seed=42):
    """
    Create training, validation and test set

    Parameters
    ----------
    valid_fraction: float between 0 and 1
        amount of data of the training cohort that can be used for
        validation instead (e.g. hyperparameter tuning) and is removed
        from the training ids.
    seed: int
        the random seed used for selecting the validation ids
    """
    dktk_ids = exclude_duplicate_dktk_ids(dktk_ids)

    train_ids = [pat_id for pat_id in dktk_ids if pat_id.startswith("DKTK")]

    n_train = len(train_ids)

    n_valid = int(n_train * valid_fraction)
    # print("using", n_valid, "patients for validation")
    np.random.seed(seed)
    valid_ids = np.random.choice(train_ids, size=n_valid, replace=False)

    # from the train_ids, use valid_fraction random samples for validation
    # and remove them from training
    train_ids = set(train_ids) - set(valid_ids)
    print("{}/{} patients for training".format(len(train_ids), len(dktk_ids)))

    print("{}/{} patients for validation".format(len(valid_ids), len(dktk_ids)))

    test_ids = list(set(dktk_ids) - set(train_ids) - set(valid_ids))
    print("{}/{} patients for test".format(len(test_ids), len(dktk_ids)))

    return train_ids, valid_ids, test_ids


def stratify_three_way(ids, train_fraction, valid_fraction, test_fraction):

    assert train_fraction + valid_fraction + test_fraction == 1.
    assert train_fraction >= 0 and valid_fraction >= 0 and test_fraction >= 0

    total = len(ids)
    n_train = round(total * train_fraction)
    n_valid = round(total * valid_fraction)
    n_test = total - (n_train + n_valid)

    # the first samples are taken for training
    train_ids = ids[:n_train]
    valid_ids = ids[n_train:(n_train+n_valid)]
    test_ids = ids[(n_train+n_valid):]

    # should be empty
    assert len(set(train_ids).intersection(valid_ids)) == 0
    # should be empty
    assert len(set(train_ids).intersection(test_ids)) == 0
    # should be empty
    assert len(set(valid_ids).intersection(test_ids)) == 0

    print("{}/{} patients for training".format(len(train_ids), total))

    print("{}/{} patients for validation".format(len(valid_ids), total))

    print("{}/{} patients for test".format(len(test_ids), total))

    return train_ids, valid_ids, test_ids


def cross_validation_splits(ids, k=10):
    """
    Create k splits of the data into training and validation
    """
    # array of arrays
    splits = np.array_split(ids, k)

    res = [None] * k
    for i in range(k):
        validation = list(splits[i])
        # exclude
        training = list(chain.from_iterable(splits[:i] + splits[i+1:]))
        res[i] = (training, validation)

    return res


def repeated_stratified_cv_splits(ids, events, cv_k, cv_reps, seed=42):
    rskf = RepeatedStratifiedKFold(
        n_splits=cv_k, n_repeats=cv_reps, random_state=seed)

    split_gen = rskf.split(ids, events)

    folds = [None] * cv_reps
    for r in range(cv_reps):
        fold = [None] * cv_k
        for k in range(cv_k):
            train_idx, valid_idx = next(split_gen)
            id_t = ids[train_idx]
            id_v = ids[valid_idx]

            fold[k] = (id_t, id_v)
        folds[r] = fold

    return folds