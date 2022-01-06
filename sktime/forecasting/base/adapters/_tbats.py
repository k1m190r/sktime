# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for using tbats forecasters in sktime framework."""

__author__ = ["mloning", "aiwalter"]
__all__ = ["_TbatsAdapter"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.utils.validation import check_n_jobs
from sktime.utils.validation.forecasting import check_sp


class _TbatsAdapter(BaseForecaster):
    """Base class for interfacing tbats forecasting algorithms."""

    _tags = {
        "ignores-exogeneous-X": True,
        "capability:pred_int": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        # "capability:predict_quantiles": True,
    }

    def __init__(
        self,
        use_box_cox=None,
        box_cox_bounds=(0, 1),
        use_trend=None,
        use_damped_trend=None,
        sp=None,
        use_arma_errors=True,
        show_warnings=True,
        n_jobs=None,
        multiprocessing_start_method="spawn",
        context=None,
    ):

        self.use_box_cox = use_box_cox
        self.box_cox_bounds = box_cox_bounds
        self.use_trend = use_trend
        self.use_damped_trend = use_damped_trend
        self.sp = sp
        self.use_arma_errors = use_arma_errors
        self.show_warnings = show_warnings
        self.n_jobs = n_jobs
        self.multiprocessing_start_method = multiprocessing_start_method
        self.context = context
        # custom sktime args
        self._forecaster = None
        self._yname = None  # .fit(y) -> y.name

        super(_TbatsAdapter, self).__init__()

    def _instantiate_model(self):
        n_jobs = check_n_jobs(self.n_jobs)
        sp = check_sp(self.sp, enforce_list=True)

        return self._ModelClass(
            use_box_cox=self.use_box_cox,
            box_cox_bounds=self.box_cox_bounds,
            use_trend=self.use_trend,
            use_damped_trend=self.use_damped_trend,
            seasonal_periods=sp,
            use_arma_errors=self.use_arma_errors,
            show_warnings=self.show_warnings,
            n_jobs=n_jobs,
            multiprocessing_start_method=self.multiprocessing_start_method,
            context=self.context,
        )

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables (ignored)

        Returns
        -------
        self : returns an instance of self.
        """
        self._forecaster = self._instantiate_model()
        self._forecaster = self._forecaster.fit(y)
        self._yname = y.name

        return self

    def _predict(self, fh, X, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : (default=None)
            NOT USED BY TBATS
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float, optional (default=0.05)
            Interpreted as "Confidence Interval" = 1 - alpha

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            Prediction intervals
        """
        if return_pred_int:
            return self._tbats_forecast_with_interval(fh, alpha)
        else:
            return self._tbats_forecast(fh)

    def _tbats_forecast(self, fh):
        """TBATS forecast without confidence interval.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon

        Returns
        -------
        y_pred : pd.Series
            Prediction
        """
        fh = fh.to_relative(cutoff=self.cutoff)

        if not fh.is_all_in_sample(cutoff=self.cutoff):
            fh_out = fh.to_out_of_sample(cutoff=self.cutoff)
            steps = fh_out.to_pandas().max()
            y_out = self._forecaster.forecast(steps=steps, confidence_level=None)

        else:
            y_out = nans(len(fh))

        # y_pred combine in and out samples
        y_in_sample = pd.Series(self._forecaster.y_hat)
        y_out_sample = pd.Series(y_out)
        y_pred = self._get_y_pred(y_in_sample=y_in_sample, y_out_sample=y_out_sample)
        y_pred.name = self._yname

        return y_pred

    def _tbats_forecast_with_interval(self, fh, conf_lev):
        """TBATS forecast with confidence interval.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        conf_lev : float
            confidence_level for TBATS

        Returns
        -------
        y_pred : pd.Series
            Prediction
        y_pred_int : pd.DataFrame
            Prediction intervals
        """
        fh = fh.to_relative(cutoff=self.cutoff)
        len_fh = len(fh)

        if not fh.is_all_in_sample(cutoff=self.cutoff):
            fh_out = fh.to_out_of_sample(cutoff=self.cutoff)
            steps = fh_out.to_pandas().max()
            _, tbats_ci = self._forecaster.forecast(
                steps=steps, confidence_level=conf_lev
            )
            out = pd.DataFrame(tbats_ci)
            y_out = out["mean"]  # aka tbats y_hat out of sample

            # pred_int
            lower = pd.Series(out["lower_bound"])
            upper = pd.Series(out["upper_bound"])
            pred_int = self._get_pred_int(lower=lower, upper=upper)

            if len(fh) != len(fh_out):
                epred_int = pd.DataFrame({"lower": nans(len_fh), "upper": nans(len_fh)})
                epred_int.index = fh.to_absolute(self.cutoff)

                in_pred_int = epred_int.index.isin(pred_int.index)
                epred_int[in_pred_int] = pred_int
                pred_int = epred_int

        else:
            y_out = nans(len_fh)
            pred_int = pd.DataFrame({"lower": nans(len_fh), "upper": nans(len_fh)})
            pred_int.index = fh.to_absolute(self.cutoff)

        # y_pred
        y_in_sample = pd.Series(self._forecaster.y_hat)
        y_out_sample = pd.Series(y_out)
        y_pred = self._get_y_pred(y_in_sample=y_in_sample, y_out_sample=y_out_sample)
        y_pred.name = self._yname

        return y_pred, pred_int

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return quantile forecasts.

        If alpha is iterable, multiple quantiles will be calculated.

        In-sample is set to NaNs, ince TBATS does not support it.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon, default = y.index (in-sample forecast)
        X : (default=None)
            NOT USED BY TBATS
        alpha : float or list of float, optional (default=[0.05, 0.95])
            A probability or list of, at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        if isinstance(alpha, (int, float)):
            alpha = [alpha]
        # else assume iterative/ list

        req_quant = np.asarray(alpha)  # requested quantiles

        # accumulator of results
        acc = pd.DataFrame([], fh.to_absolute(self.cutoff))

        for q1 in req_quant:

            if q1 in acc.columns:
                # skip as this quantile is already computed by tbats
                continue

            # q = 0.5 -> conf_lev = 0 -> y_pred = pred_int[lower] = pred_int[upper]
            # so don't compute CI intervals simply save y_hat = predictions
            if q1 == 0.5:
                acc[q1] = self._tbats_forecast(fh)
                continue

            # otherwise compute CI intervals

            # tbats with CI intervals
            conf_level = np.abs(1 - q1 * 2)
            y_pred, pred_int = self._tbats_forecast_with_interval(fh, conf_level)

            # preserve q = 0.5, which is calculated and returned anyway
            acc[0.5] = y_pred

            q2 = 1 - q1  # the other quantile

            # _get_pred_int() returns DataFrame with "lower" "upper" columns
            # rename them to quantiles
            colnames = {
                "lower": q1 if q1 < q2 else q2,
                "upper": q2 if q2 > q1 else q1,
            }
            pred_int = pred_int.rename(columns=colnames)

            # add to acc
            for q in [q1, q2]:
                acc[q] = pred_int[q]

        # order as requested and drop un-requested
        quantiles = acc.reindex(columns=req_quant)

        # the y.name for multi-index or "quantiles"
        col_name = "quantiles" if (self._yname in {None, ""}) else self._yname
        quantiles.columns = pd.MultiIndex.from_product([[col_name], req_quant])

        return quantiles

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        self.check_is_fitted()
        fitted_params = {}
        for name in self._get_fitted_param_names():
            fitted_params[name] = getattr(self._forecaster, name, None)
        return fitted_params

    def _get_fitted_param_names(self):
        """Get names of fitted parameters."""
        return self._fitted_param_names

    def _get_pred_int(self, lower, upper):
        """Combine lower/upper bounds of pred.intervals, slice on fh.

        Parameters
        ----------
        lower : pd.Series
            Lower bound (can contain also in-sample bound)
        upper : pd.Series
            Upper bound (can contain also in-sample bound)

        Returns
        -------
        pd.DataFrame
            pred_int, prediction intervals (out-sample, sliced by fh)
        """
        pred_int = pd.DataFrame({"lower": lower, "upper": upper})
        # Out-sample fh
        fh_out = self.fh.to_out_of_sample(cutoff=self.cutoff)
        # If pred_int contains in-sample prediction intervals
        if len(pred_int) > len(self._y):
            len_out = len(pred_int) - len(self._y)
            # Workaround for slicing with negative index
            pred_int["idx"] = [x for x in range(-len(self._y), len_out)]
        # If pred_int does not contain in-sample prediction intervals
        else:
            pred_int["idx"] = [x for x in range(len(pred_int))]
        pred_int = pred_int.loc[
            pred_int["idx"].isin(fh_out.to_indexer(self.cutoff).values)
        ]
        pred_int.index = fh_out.to_absolute(self.cutoff)
        pred_int = pred_int.drop(columns=["idx"])
        return pred_int


def nans(length):
    """Return l NaNs."""
    return np.full(length, np.nan)
