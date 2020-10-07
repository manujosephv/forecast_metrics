from gluonts.evaluation import Evaluator
import numpy as np
import pandas as pd
from gluonts.model.forecast import Forecast, Quantile
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    Callable,
)
import logging


class IntermittentEvaluator(Evaluator):

    default_quantiles = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

    def __init__(
        self,
        quantiles: Iterable[Union[float, str]] = default_quantiles,
        seasonality: Optional[int] = None,
        alpha: float = 0.05,
        calculate_owa: bool = False,
        num_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        median: Optional[bool] = True,
        calculate_spec: Optional[bool] = False
    ) -> None:
        self.median = median
        self.calculate_spec = calculate_spec

        super().__init__(
            quantiles, seasonality, alpha, calculate_owa, num_workers, chunk_size
        )

    # Hacking the seasonal error method to return Naive Fcst Error
    @staticmethod
    def naive_fc(
        time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast
    ) -> np.ndarray:
        """

        Parameters
        ----------
        time_series
        forecast

        Returns
        -------
        np.ndarray
            time series without the forecast dates
        """

        assert forecast.index.intersection(time_series.index).equals(forecast.index), (
            "Index of forecast is outside the index of target\n"
            f"Index of forecast: {forecast.index}\n Index of target: {time_series.index}"
        )

        # Remove the prediction range
        # If the prediction range is not in the end of the time series,
        # everything after the prediction range is truncated
        date_before_forecast = forecast.index[0] - forecast.index[0].freq
        naive_fc = time_series.loc[date_before_forecast].values.item()
        return np.atleast_1d(
            np.squeeze(pd.Series(index=forecast.index, data=naive_fc).transpose())
        )

    # def naive_fc(self, pred_length, past_data: np.ndarray) -> float:
    #     r"""
    #     .. math::

    #         seasonal_error = mean(|Y[t] - Y[t-1]|)

    #     where m is the seasonal frequency
    #     https://www.m4.unic.ac.cy/wp-content/uploads/2018/03/M4-Competitors-Guide.pdf
    #     """
    #     forecast_freq = 1
    #     y_tm = past_data[-(pred_length + 1): -forecast_freq]
    #     return y_tm

    @staticmethod
    def signed_error(target, forecast):
        return np.sum(target - forecast)

    @staticmethod
    def cum_error(target, forecast):
        return np.cumsum(target - forecast)

    @staticmethod
    def nos_p(target, forecast):
        c = np.cumsum(target - forecast)
        mask = target > 0
        return np.sum(c[mask] > 0) / np.sum(mask)

    @staticmethod
    def pis(target, forecast):
        cfe_t = np.cumsum(target - forecast)
        return np.sum(-1 * cfe_t)

    @staticmethod
    def mae(target, forecast):
        return np.mean(np.abs(target - forecast))

    @staticmethod
    def mean_relative_abs_error(target, forecast, seasonal_error):
        return np.mean(np.abs(target - forecast) / np.abs(target - seasonal_error))

    @staticmethod
    def percent_better(target, forecast, seasonal_error):
        mae = np.abs(target - forecast)
        mae_star = np.abs(target - seasonal_error)
        pb = mae > mae_star
        return np.sum(pb) / len(pb)
    
    @staticmethod
    #https://github.com/DominikMartin/spec_metric/blob/master/spec_metric/_metric.py
    def spec(y_true, y_pred, a1=0.75, a2=0.25):
        """Stock-keeping-oriented Prediction Error Costs (SPEC)
        Read more in the :ref:`https://arxiv.org/abs/2004.10537`.
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground truth (correct) target values.
        y_pred : array-like of shape (n_samples,)
            Estimated target values.
        a1 : opportunity costs weighting parameter
            a1 ∈ [0, ∞]. Default value is 0.75.
        a2 : stock-keeping costs weighting parameter
            a2 ∈ [0, ∞]. Default value is 0.25.
        Returns
        -------
        loss : float
            SPEC output is non-negative floating point. The best value is 0.0.
        Examples
        --------
        >>> from spec_metric import spec
        >>> y_true = [0, 0, 5, 6, 0, 5, 0, 0, 0, 8, 0, 0, 6, 0]
        >>> y_pred = [0, 0, 5, 6, 0, 5, 0, 0, 8, 0, 0, 0, 6, 0]
        >>> spec(y_true, y_pred)
        0.1428...
        >>> spec(y_true, y_pred, a1=0.1, a2=0.9)
        0.5142...
        """
        assert len(y_true) > 0 and len(y_pred) > 0
        assert len(y_true) == len(y_pred)

        sum_n = 0
        for t in range(1, len(y_true) + 1):
            sum_t = 0
            for i in range(1, t + 1):
                delta1 = np.sum([y_k for y_k in y_true[:i]]) - np.sum([f_j for f_j in y_pred[:t]])
                delta2 = np.sum([f_k for f_k in y_pred[:i]]) - np.sum([y_j for y_j in y_true[:t]])

                sum_t = sum_t + np.max([0, a1 * np.min([y_true[i - 1], delta1]), a2 * np.min([y_pred[i - 1], delta2])]) * (
                            t - i + 1)
            sum_n = sum_n + sum_t
        return sum_n / len(y_true)
    
    
    @staticmethod
    def maape(target, forecast):
        ape = np.zeros_like(target, dtype="float")
        mask = np.logical_not((target == 0) & (forecast == 0))
        ape = np.divide(np.abs(target - forecast), target, out=ape, where=mask)
        return np.mean(np.arctan(ape))

    def get_metrics_per_ts(
        self, time_series: Union[pd.Series, pd.DataFrame], forecast: Forecast
    ) -> Dict[str, Union[float, str, None]]:
        pred_target = np.array(self.extract_pred_target(time_series, forecast))
        pred_target = np.ma.masked_invalid(pred_target)

        # required for seasonal_error and owa calculation
        past_data = np.array(self.extract_past_data(time_series, forecast))
        past_data = np.ma.masked_invalid(past_data)
        if self.median:
            fcst = forecast.quantile(0.5)
        else:
            fcst = forecast.mean
        naive_fc = self.naive_fc(time_series, forecast)
        seasonal_error = np.abs(naive_fc - pred_target)
        seasonal_sq_error = np.square(naive_fc - pred_target)
        nonzero_mask = pred_target > 0
        metrics = {
            "item_id": forecast.item_id,
            "non_zero_n": np.sum(nonzero_mask),
            "n": len(pred_target),
            "abs_target_sum": self.abs_target_sum(pred_target),
            "abs_target_mean": self.abs_target_mean(pred_target),
            "seasonal_error": seasonal_error.sum(),
            "seasonal_sq_error": seasonal_sq_error.sum(),
            "abs_error": self.abs_error(pred_target, fcst),
            "cum_error": self.cum_error(pred_target, fcst),
            "CFE": self.signed_error(pred_target, fcst),
            "CFE_min": np.min(self.cum_error(pred_target, fcst)),
            "CFE_max": np.max(self.cum_error(pred_target, fcst)),
            "NOSp": self.nos_p(pred_target, fcst),
            "PIS": self.pis(pred_target, fcst),
            "MSE": self.mse(pred_target, fcst),
            "RMSE": np.sqrt(self.mse(pred_target, fcst)),
            "MAE": self.mae(pred_target, fcst),
            "MRAE": self.mean_relative_abs_error(pred_target, fcst, naive_fc),
            # "MSEn": self.mse(pred_target[nonzero_mask], fcst[nonzero_mask]),
            # "MAEn": self.mae(pred_target[nonzero_mask], fcst[nonzero_mask]),
            "MASE": self.mase(pred_target, fcst, seasonal_error.mean()),
            "MAPE": self.mape(pred_target, fcst),
            "RelMAE": self.mae(pred_target, fcst) / self.mae(pred_target, naive_fc),
            "RelRMSE": np.sqrt(self.mse(pred_target, fcst))
            / np.sqrt(self.mse(pred_target, naive_fc)),
            "PBMAE": self.percent_better(pred_target, fcst, naive_fc),
            "MAAPE": self.maape(pred_target, fcst),
            "sMAPE": self.smape(pred_target, fcst),
            "SPEC_0.75": self.spec(pred_target, fcst, a1=0.75, a2=0.25),
            "SPEC_0.5": self.spec(pred_target, fcst, a1=0.5, a2=0.5),
            "SPEC_0.25": self.spec(pred_target, fcst, a1=0.25, a2=0.75)
        }
        if self.calculate_spec:
            metrics["SPEC_0.75"]= self.spec(pred_target, fcst, a1=0.75, a2=0.25),
            metrics["SPEC_0.5"]= self.spec(pred_target, fcst, a1=0.5, a2=0.5),
            metrics["SPEC_0.25"]= self.spec(pred_target, fcst, a1=0.25, a2=0.75)
        for quantile in self.quantiles:
            forecast_quantile = forecast.quantile(quantile.value)

            metrics[quantile.loss_name] = self.quantile_loss(
                pred_target, forecast_quantile, quantile.value
            )
            metrics[quantile.coverage_name] = self.coverage(
                pred_target, forecast_quantile
            )

        return metrics

    def get_aggregate_metrics(
        self, metric_per_ts: pd.DataFrame
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        agg_funs = {
            "MSE": "mean",
            "MAE": "mean",
            "MRAE": "mean",
            # "MSEn": "mean",
            # "MAEn": "mean",
            "MASE": "mean",
            "MAPE": "mean",
            "MAAPE": "mean",
            "sMAPE": "mean",
            "abs_target_sum": "sum",
            "abs_target_mean": "mean",
            "abs_error": "sum",
            "seasonal_error": "mean",
            "n": "sum",
            "non_zero_n": "sum",
            "CFE": "sum",
            "PIS": "sum",
        }
        for quantile in self.quantiles:
            agg_funs[quantile.loss_name] = "sum"
            agg_funs[quantile.coverage_name] = "mean"

        assert (
            set(metric_per_ts.columns) >= agg_funs.keys()
        ), "The some of the requested item metrics are missing."

        totals = {key: metric_per_ts[key].agg(agg) for key, agg in agg_funs.items()}

        # derived metrics based on previous aggregate metrics
        totals["NOSp"] = (
            metric_per_ts["NOSp"] * metric_per_ts["non_zero_n"]
        ).sum() / metric_per_ts["non_zero_n"].sum()
        totals["PBMAE"] = (
            metric_per_ts["PBMAE"] * metric_per_ts["n"]
        ).sum() / metric_per_ts["n"].sum()
        if self.calculate_spec:
            totals["SPEC_0.75"] = (
                metric_per_ts["SPEC_0.75"] * metric_per_ts["n"]
            ).sum() / metric_per_ts["n"].sum()
            totals["SPEC_0.5"] = (
                metric_per_ts["SPEC_0.5"] * metric_per_ts["n"]
            ).sum() / metric_per_ts["n"].sum()
            totals["SPEC_0.25"] = (
                metric_per_ts["SPEC_0.25"] * metric_per_ts["n"]
            ).sum() / metric_per_ts["n"].sum()
        totals["RMSE"] = np.sqrt(totals["MSE"])
        cum_errs = np.array([row for row in metric_per_ts["cum_error"]])
        cum_errs = np.sum(cum_errs, axis=0)
        totals["CFE_min"] = cum_errs.min()
        totals["CFE_max"] = cum_errs.max()
        totals["RelMAE"] = totals["MAE"] / metric_per_ts["seasonal_error"].mean()
        totals["RelRMSE"] = totals["RMSE"] / np.sqrt(
            metric_per_ts["seasonal_sq_error"].mean()
        )

        flag = totals["abs_target_mean"] == 0
        totals["NRMSE"] = np.divide(
            totals["RMSE"] * (1 - flag), totals["abs_target_mean"] + flag
        )

        flag = totals["abs_target_sum"] == 0
        totals["ND"] = np.divide(
            totals["abs_error"] * (1 - flag), totals["abs_target_sum"] + flag
        )

        all_qLoss_names = [quantile.weighted_loss_name for quantile in self.quantiles]
        for quantile in self.quantiles:
            totals[quantile.weighted_loss_name] = np.divide(
                totals[quantile.loss_name], totals["abs_target_sum"]
            )

        totals["mean_wQuantileLoss"] = np.array(
            [totals[ql] for ql in all_qLoss_names]
        ).mean()

        totals["MAE_Coverage"] = np.mean(
            [
                np.abs(totals[q.coverage_name] - np.array([q.value]))
                for q in self.quantiles
            ]
        )
        return totals, metric_per_ts
