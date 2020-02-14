import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


from lifelines.utils import concordance_index

from .base_estimator import BaseEstimator


class RegressionModel(BaseEstimator):
    def __init__(self, model_cls, feat_cols=["Age"], time_col="Survival", id_col="id",
                 event_col=None, verbose=True, scale_features=False, **kwargs):

        super().__init__(
            model_cls=model_cls,
            feat_cols=feat_cols,
            time_col=time_col,
            event_col=event_col,
            id_col=id_col,
            verbose=verbose,
            scale_features=scale_features,
            **kwargs)

    def _convert_df_to_arrays(self, df):
        df, ids = self.remove_nans(df)

        X = df[self.feat_cols].values.reshape((-1, len(self.feat_cols)))
        try:
            y = df[self.time_col].values.astype(np.float32)
        except:
            # the time column is probably only present in the training data so we leave it at None
            print("Seems like time is not available!")
            y = None

        if self.verbose:
            print("X.shape={}, y.shape={}".format(X.shape, y.shape if y is not None else "-"))

        return X, y, ids

    def fit(self, train_df):

        X, y, _ = self._convert_df_to_arrays(train_df)
        assert y is not None

        if self.scaler is not None:
            # NOTE: do we want to "remember" the info on the training set and apply
            # to test set or do we want to scale the test data individually to (0,1)
            # without applying the min/max of the training set?
            X = self.scaler.fit_transform(X)
            if self.verbose:
                print("Transformed features to (0,1) range before fitting!")

        self.model = self.model.fit(X, y)

        if self.verbose:
            print("model score:", self.model.score(X, y))
            print("model_intercept:", self.model.intercept_)
            print("model_coefficients:", self.model.coef_)

    def _predict(self, pred_df):
        X, y, ids = self._convert_df_to_arrays(pred_df)

        if self.scaler is not None:
            X = self.scaler.transform(X)
            if self.verbose:
                print("Scaled features based on training set results!")

        y_pred = self.model.predict(X)

        pred = pd.DataFrame({
            self.id_col: ids,
            'pred_per_pat': y_pred
        })

        if y is not None:
            # we can append some information to the pred_df as well
            pred["truth"] = y
            pred["error"] = y - y_pred

        return pred

    def _performance(self, pred):
        raise NotImplementedError

    def predict(self, pred_df):
        pred = self._predict(pred_df)

        perf = self._performance(pred)

        return pred, perf


class LinearRegressionModel(RegressionModel):
    def __init__(self, feat_cols=["Age"], time_col="Survival", id_col="id",
                 event_col=None, verbose=True, scale_features=False, **kwargs):

        super().__init__(
            model_cls=LinearRegression,
            feat_cols=feat_cols,
            time_col=time_col,
            event_col=event_col,
            id_col=id_col,
            verbose=verbose,
            scale_features=scale_features,
            **kwargs)

    def _performance(self, pred):
        perf = None
        if "truth" in pred.columns:
            # only if we have the true label we can return performance
            # values
            diff = pred["truth"] - pred["error"]
            perf = pd.DataFrame({
                'MSE': [np.mean(diff**2)],
                'MAE': [np.mean(np.abs(diff))],
                'C-index': [concordance_index(
                        pred["truth"].values, pred["pred_per_pat"].values)],
            })
        return perf
