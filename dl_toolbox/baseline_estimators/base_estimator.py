from sklearn.preprocessing import MinMaxScaler


class BaseEstimator(object):
    def __init__(self, model_cls, feat_cols=["Age"], time_col="Survival", id_col="id",
                 event_col=None,
                 verbose=True, scale_features=False, **kwargs):
        self.feat_cols = feat_cols
        self.time_col = time_col
        self.event_col = event_col
        self.verbose = verbose
        self.id_col = id_col

        self.scaler = MinMaxScaler(feature_range=(0, 1)) if scale_features else None

        self.model = model_cls(**kwargs)

    def remove_nans(self, df):
        # remove nan rows from df
        len_ori = len(df)
        df = df.dropna(axis=0, how="any")
        len_nona = len(df)
        if len_ori != len_nona:
            print("Warning: data contained nan. {} rows were removed!".format(
                len_ori - len_nona))

        # the ids left over after possibly removing nan values
        ids = df[self.id_col].values

        return df, ids

    def fit(self, train_df):
        raise NotImplementedError

    def predict(self, pred_df):
        """
        Returns
        -------
        pred_df: pd.DataFrame with predictions
        perf_df: pd.DataFrame with performance metrics of the prediction
                 if labels are provided, None otherwise
        """
        raise NotImplementedError