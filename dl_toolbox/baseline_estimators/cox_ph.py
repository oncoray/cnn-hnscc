
import pandas as pd
import numpy as np

from lifelines import CoxPHFitter
from lifelines.utils import qth_survival_times, concordance_index

from .base_estimator import BaseEstimator


class CoxModel(BaseEstimator):
    def __init__(self, feat_cols=["volume"], time_col="Survival", id_col="id",
                 event_col=None, verbose=True, **kwargs):
        super().__init__(
            model_cls=CoxPHFitter,
            feat_cols=feat_cols,
            time_col=time_col,
            event_col=event_col,
            id_col=id_col,
            verbose=verbose,
            **kwargs)

    def _prepare_df(self, df):
        df, ids = self.remove_nans(df)
        # convert the id column to index (no longer treated as a column)
        df = df.set_index(self.id_col)

        # use only the time col and the selected features
        all_cols = list(df.columns)

        wanted_cols = [c for c in self.feat_cols]  # copy and not direct assignment as reference!
        if self.time_col in all_cols:
            wanted_cols.append(self.time_col)

        if self.event_col is None or self.event_col not in all_cols:
            print("Event column not present in df or not specified. Will assume all patients have events!")
            # add a column of all ones to the dataframe
            self.event_col = "event_col"
            df[self.event_col] = 1.

        wanted_cols.append(self.event_col)

        drop_cols = list(set(all_cols) - set(wanted_cols))
        print("drop_cols =", drop_cols)
        df = df.drop(drop_cols, axis=1)

        return df, ids

    def fit(self, train_df):

        train_df, _ = self._prepare_df(train_df)

        if self.scaler is not None:
            # NOTE: do we want to "remember" the info on the training set and apply
            # to test set or do we want to scale the test data individually to (0,1)
            # without applying the min/max of the training set?
            X = train_df[self.feat_cols].values
            X_scaled = self.scaler.fit_transform(X)
            train_df[self.feat_cols] = X_scaled

            if self.verbose:
                print("Transformed features to (0,1) range before fitting!")

        try:
            self.model = self.model.fit(
                train_df, duration_col=self.time_col, event_col=self.event_col,
                show_progress=self.verbose)
            if self.verbose:
                self.model.print_summary()

        except Exception as e:
            print("[W]: fitting Cox model failed: {}".format(e))
            raise e

    def predict(self, df, q=0.5):
        df, _ = self._prepare_df(df)

        if self.scaler is not None:
            X = df[self.feat_cols].values
            X_scaled = self.scaler.transform(X)
            df[self.feat_cols] = X_scaled
            if self.verbose:
                print("Scaled features based on training set results!")

        # the cox model predicts log hazards and from that can also compute individual survival time curves
        surv_funs = self.model.predict_survival_function(df)
        # and we take the time point were the curve reaches 0.5
        pred_time = qth_survival_times(q, surv_funs).squeeze()

        # print("head(df) =\n{}".format(df.head(5)))
        log_hazards = self.model.predict_log_partial_hazard(df).values.squeeze()
        # print("Log_hazards = {}".format(log_hazards.shape))
        # print("pred_time.shape", pred_time.shape)
        # print("ids.shape", df[self.id_col].values.shape)

        pred_df = pd.DataFrame({
            self.id_col: df.index.values,
            'pred_time': pred_time,
            'pred_per_pat(log_hazard)': log_hazards
        })

        perf_df = None
        # in case we have labels
        if self.time_col in df.columns and self.event_col in df.columns:
            # we can append some information to the pred_df as well
            true_time = df[self.time_col].values
            event_status = df[self.event_col].values
            diff = true_time - pred_time

            pred_df[self.event_col + "_truth"] = event_status
            pred_df[self.time_col + "_truth"] = true_time
            pred_df["error(time_prediction)"] = diff

            # only if we have the true label we can return performance
            # values
            perf_df = pd.DataFrame({
                'MSE': [np.mean(diff**2)],
                'MAE': [np.mean(np.abs(diff))],
                'C-index_time': [concordance_index(true_time, pred_time, event_status)],
                'C-index_log_hazard': [concordance_index(true_time, log_hazards, event_status)]
            })

        return pred_df, perf_df
