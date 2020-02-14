import numpy as np
import pandas as pd
import collections

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from .linear_regression import RegressionModel


class LogisticRegressionModel(RegressionModel):
    def __init__(self, time_thresholds, feat_cols=["Age"], time_col="Survival", id_col="id",
                 event_col=None, verbose=True, scale_features=False, **kwargs):

        if not isinstance(time_thresholds, collections.Iterable):
            time_thresholds = [time_thresholds]
        self.time_thresholds = np.array(time_thresholds)

        super().__init__(
            model_cls=LogisticRegression,
            feat_cols=feat_cols,
            time_col=time_col,
            event_col=event_col,
            id_col=id_col,
            verbose=verbose,
            scale_features=scale_features,
            **kwargs)

    def _convert_df_to_arrays(self, df):
        X, y, ids = super()._convert_df_to_arrays(df)

        # binarize the time into classes based on the thresholds
        if y is not None:
            y = np.digitize(y, self.time_thresholds)

        return X, y, ids

    def _performance(self, pred):
        perf = None
        if "truth" in pred.columns:
            # compute AUC
            y_true = pred["truth"].values
            y_pred = pred["pred_per_pat"].values

            auc = roc_auc_score(y_true, y_pred)

            # classification metrics
            prec, recall, f1_score, _ = precision_recall_fscore_support(
                y_true, y_pred)

            perf = pd.DataFrame({
                'AUC': [auc],
                'precision': [prec],
                'recall': [recall],
                'F1-score': [f1_score]
            })

        return perf




