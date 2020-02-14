import os
import pandas as pd
import numpy as np

from inference import _subdirectories_full_path
from dl_toolbox.data.read import read_baseline_feats

from scipy.stats import mannwhitneyu

# pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 300)

def list_nan_for_each_column(df, id_col="ID_Radiomics"):
    for c in df.columns:
        nan_samples = df[df[c].isna()]
        n_nan = len(nan_samples)
        id_nan = sorted(nan_samples[id_col].values)
        print("\n{} number of na={}, ids={}".format(c, n_nan, id_nan))


def cohort_characteristics(clinical_df, train_ids, id_col="ID_Radiomics"):
    def print_range(pd_col, tag):
        nan_samples = pd_col.isna()
        n_nan = len(pd_col[nan_samples])
        nan_pctg = n_nan / len(pd_col) * 100
        vals = pd_col.dropna().values

        median = np.median(vals)
        m = np.min(vals)
        M = np.max(vals)
        print("\t {}: {:.2f} ({:.2f}, {:.2f}), unknown={}, (%={:.2f})".format(tag, median, m, M, n_nan, nan_pctg))

    def print_counts(pd_col, tag):
        nan_samples = pd_col.isna()
        n_nan = len(pd_col[nan_samples])
        nan_pctg = n_nan / len(pd_col) * 100
        vals = pd_col.dropna().values

        absolute = pd_col.value_counts(dropna=False, ascending=True)
        relative = pd_col.value_counts(dropna=False, ascending=True, normalize=True) * 100

        combined = pd.concat([absolute, relative], axis=1)
        combined.columns = ["absolute", "relative"]

        print("\t{}:\n{}".format(tag, combined))

    full = clinical_df.copy()
    print(full.columns)
    # fuse site columns into a single one (undoing the one-hot encoding)
    site_cols = ["Oropharynx", "Larynx", "Hypopharynx", "Oral cavity"]
    full["site"] = full[site_cols].idxmax(1)
    full.drop(site_cols, axis=1, inplace=True)
    # remove incomplete tumor volume columns
    full.drop(["GTVtu", "ln(GTVtu)", "ln(GTVtu_from_mask)"], axis=1, inplace=True)
    # maybe do the test in python
    full.to_csv("/home/MED/starkeseb/cohort_analysis.csv", index=False)
    # remove cohort column
    full.drop(["cohort"], axis=1, inplace=True)
    # print(full)

    train = full[full[id_col].isin(train_ids)]
    test = full[~full[id_col].isin(train_ids)]
    print("for cohort checking", train.shape, test.shape)
    print(full)

    for col in full.columns:
        if col == id_col:
            continue
        col_train = train[col].dropna()
        col_test = test[col].dropna()
        print("\n", col)
        if col in ["LRCtime", "Age", "GTVtu_from_mask"]:
            # continuous column
            # compute pvalue through man-whitney u
            pval = mannwhitneyu(col_train.values.squeeze(), col_test.values.squeeze(), use_continuity=True, alternative="two-sided").pvalue
            # print range
            print_range(col_train, tag="exploratory")
            print_range(col_test, tag="test")
        else:
            # categorical column
            # compute pvalue through chi-squared test: don't know if this is implemented in python
            pval = np.nan
            # print counts
            print_counts(col_train, tag="exploratory")
            print_counts(col_test, tag="test")

        print("\n\tp-value=", pval)


if __name__ == "__main__":

    # read the ids that we use of the DKTK cohort
    dktk_ids = pd.read_csv("/home/MED/starkeseb/dktk_ids.csv", header=None).values.squeeze()
    dktk_train_ids = pd.read_csv("/home/MED/starkeseb/dktk_train_ids.csv", header=None).values.squeeze()
    dktk_test_ids = np.array(list(set(dktk_ids) - set(dktk_train_ids)))
    print(dktk_ids.shape, dktk_train_ids.shape, dktk_test_ids.shape)
    # print(len(dktk_ids))
    # print("dktk_ids\n", dktk_ids)

    # read the outcome file
    # outcome_df = pd.read_csv("/home/MED/starkeseb/git/HNSCC_clinical_data/HNSCC_clinical.csv",
    #                          sep=";", decimal=",",
    #                         #  lineterminator=";"
    #                          )
    # is a dictionary with multiple sheets
    outcome_dict = pd.read_excel("/home/MED/starkeseb/git/HNSCC_clinical_data/HNSCC_clinical.xlsx",
                               sheet_name=None)
    outcome_df = outcome_dict["Daten"]
    study_df = outcome_dict["Studien"]

    # the patients of our study, sanity check the both sources of IDs to be the same
    radiomics_ids = study_df.loc[study_df["Radiomics2019"] == 1, "ID_Radiomics"].values
    assert np.all(sorted(dktk_ids) == sorted(radiomics_ids))

    radiomics_outcome = outcome_df[outcome_df["ID_Radiomics"].isin(radiomics_ids)]

    # select the features we need
    radiomics_outcome = radiomics_outcome[["ID_Radiomics", "LRC", "LRCtime",
                                           "Gender", "Age", "Oral cavity", "Oropharynx",
                                           "Hypopharynx", "Larynx", "UICC2010", "T", "N",
                                           "Grading", "p16", "GTVtu"]]
    #### insert a cohort column for training vs testing for FAMILIAR
    radiomics_outcome["cohort"] = "test"
    radiomics_outcome.loc[radiomics_outcome["ID_Radiomics"].isin(dktk_train_ids), "cohort"] = "training"
    print(len(radiomics_outcome[radiomics_outcome["cohort"] == "training"]))

    # since we later want to use ln(GTVtu) in the models
    # here we have volume measured in cm^3
    radiomics_outcome["ln(GTVtu)"] = np.log(radiomics_outcome["GTVtu"])
    # print(radiomics_outcome)

    # since not all tumor volumes are present there we read the tumor volumes from our
    # baseline features where we computed it by summing over the masks
    data_path_base = "/home/MED/starkeseb/mbro_local/data/DKTK/"
    data_subdirs = _subdirectories_full_path(data_path_base)  # the subcohorts
    baseline_files = [os.path.join(d, "numpy_preprocessed", "baseline_features.csv") for d in data_subdirs]
    baseline_df = read_baseline_feats(baseline_files)
    volume_df = baseline_df[["id", "volume"]]  # this is still in mm^3
    volume_df["volume"] /= 1000.  # convert to cm^3

    volume_df.rename(columns={'volume': 'GTVtu_from_mask', 'id': 'ID_Radiomics'}, inplace=True)
    volume_df["ln(GTVtu_from_mask)"] = np.log(volume_df["GTVtu_from_mask"])

    # join this to the outcome so we can later decide which volume to use
    radiomics_outcome = pd.merge(radiomics_outcome, volume_df, on="ID_Radiomics")

    # check how many missing values per column
    list_nan_for_each_column(radiomics_outcome)

    #### grouping of subcategories
    # for T stage we have to get rid of something like 4a and map it to 4
    # print(radiomics_outcome["T"].values)
    radiomics_outcome.loc[radiomics_outcome["T"] == "4a", "T"] = 4
    radiomics_outcome.loc[radiomics_outcome["T"] == "4b", "T"] = 4
    # for N stage we have 2a, 2b and 2c
    # print(radiomics_outcome["N"].values)
    radiomics_outcome.loc[radiomics_outcome["N"] == "2a", "N"] = 2
    radiomics_outcome.loc[radiomics_outcome["N"] == "2b", "N"] = 2
    radiomics_outcome.loc[radiomics_outcome["N"] == "2c", "N"] = 2

    # show the characteristics before we start imputing and manipulating columns
    cohort_characteristics(radiomics_outcome, dktk_train_ids)


    #### imputation of missing values based on the training part and application to test part
    radiomics_train = radiomics_outcome[radiomics_outcome["ID_Radiomics"].isin(dktk_train_ids)]
    # radiomics_test = radiomics_outcome[radiomics_outcome["ID_Radiomics"].isin(dktk_test_ids)]
    # print(radiomics_train.shape, radiomics_test.shape)
    for col in ["Gender", "Age", "N", "Grading"]:

        if col == "Age":
            # for age we have to impute the median
            impute_value = np.median(radiomics_train[col].dropna().values)
        else:
            # the remaining columns are categorical and we use the most often occuring value
            # on the training set
            value_counts = radiomics_train[col].dropna().value_counts()
            impute_value = value_counts.idxmax()
            print("{}: value counts:\n{}:".format(col, value_counts))

        print("{}: impute_value={}".format(col, impute_value))

        impute_col = col + "_impute"
        radiomics_outcome[impute_col] = radiomics_outcome[col]
        rows_with_na = radiomics_outcome[col].isna()
        radiomics_outcome.loc[rows_with_na, impute_col] = impute_value

        # remove the original column and rename the "_impute" column
        radiomics_outcome.drop([col], axis=1, inplace=True)
        radiomics_outcome.rename({impute_col: col}, axis=1, inplace=True)

    # check again that we dont have nan anymore
    list_nan_for_each_column(radiomics_outcome, id_col="ID_Radiomics")

    #### binarization of categorical features
    radiomics_outcome['UICC2010<4'] = (radiomics_outcome['UICC2010'] < 4).astype(int)
    # print(radiomics_outcome[["ID_Radiomics", "UICC2010", "UICC2010<4"]])
    radiomics_outcome["T<4"] = (radiomics_outcome["T"] < 4).astype(int)
    # print(radiomics_outcome[["ID_Radiomics", "T", "T<4"]])
    radiomics_outcome["N<2"] = (radiomics_outcome["N"] < 2).astype(int)
    # print(radiomics_outcome[["ID_Radiomics", "N", "N<2"]])
    radiomics_outcome["Grading<2"] = (radiomics_outcome["Grading"] < 2).astype(int)
    print(radiomics_outcome[["ID_Radiomics", "Grading", "Grading<2"]])
    # for p16 we make two features: unknown and p16pos
    p16_unknown = radiomics_outcome["p16"].isna()
    radiomics_outcome["p16_unknown"] = (p16_unknown).astype(int)
    radiomics_outcome["p16_pos"] = 0
    radiomics_outcome.loc[~p16_unknown, "p16_pos"] = radiomics_outcome.loc[~p16_unknown, "p16"]
    # print(radiomics_outcome[["ID_Radiomics", "p16", "p16_unknown", "p16_pos"]])

    #### normalization based on the training part and application to the test part
    # for Age and ln(GTVtu_from_mask)
    for col in ["Age", "ln(GTVtu_from_mask)"]:
        values = radiomics_train[col].values
        train_mu = np.mean(values)
        train_sigma = np.std(values)
        print("{}: train_mu={}, train_sigma={}".format(col, train_mu, train_sigma))

        normalize_col = col + "_zscore"
        radiomics_outcome[normalize_col] = radiomics_outcome[col]
        radiomics_outcome[normalize_col] -= train_mu
        if train_sigma > 0:
            radiomics_outcome[normalize_col] /= train_sigma

        # remove the original column and rename the "_zscore" column
        # radiomics_outcome.drop([col], axis=1, inplace=True)
        # radiomics_outcome.rename({normalize_col: col}, axis=1, inplace=True)


    radiomics_outcome.drop([
        "Age", "ln(GTVtu_from_mask)",             # those were the non-zscore versions
        "GTVtu", "ln(GTVtu)", "GTVtu_from_mask",  # we only keep the zscore normalized variant of log-volume
        "UICC2010", "T", "N", "Grading", "p16",   # not binarized
        ], axis=1, inplace=True)
    print("\nRadiomics features for FAMILIAR\ncolumns={}\n{}".format(radiomics_outcome.columns, radiomics_outcome))
    radiomics_outcome.to_csv("/home/MED/starkeseb/g40fs4-hprt/HPRT-Data/ONGOING_PROJECTS/"
                             "Radiomics/RadiomicsAnalysis/Project/DeepRadiomics_Sebastian/"
                             "Experiments/paper_evaluation_of_dl_approaches/clinical_model/clinical_features.csv",
                             index=False)

    # Now also write the outcome into the DKTK subdirs so we can use it
    # as clinical features for the neural networks
    clinical_features = radiomics_outcome.copy()
    clinical_features.rename({'ID_Radiomics': 'id'}, axis=1, inplace=True)
    clinical_features.drop(["LRC", "LRCtime", "cohort"], axis=1, inplace=True)

    # we only want volume since this was also only relevant feature in clinical model
    clinical_features = clinical_features[["id", "ln(GTVtu_from_mask)_zscore"]]

    print("\nClinical features for DL models\ncolumns={}\n{}".format(clinical_features.columns, clinical_features))

    # check again that we dont have nan anymore
    list_nan_for_each_column(clinical_features, id_col="id")

    for subdir in data_subdirs:
        data_dir = os.path.join(subdir, "numpy_preprocessed")
        patients = [os.path.basename(d) for d in _subdirectories_full_path(data_dir)]

        clin_df = clinical_features[clinical_features.id.isin(patients)]
        # print(subdir, patients, "\n", clin_df)
        target_file = os.path.join(data_dir, "clinical_features.csv")
        clin_df.to_csv(target_file, index=False)