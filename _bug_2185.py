# -*- coding: utf-8 -*-
"""Bug 2185."""
# %%
from warnings import filterwarnings

from sktime.datasets import load_shampoo_sales
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.transformations.series.impute import Imputer

filterwarnings("ignore", category=FutureWarning)

y = load_shampoo_sales()
fh = [1, 2, 3]
cv = ExpandingWindowSplitter(start_with_window=True, fh=fh)
forecaster = NaiveForecaster()
param_grid = {"strategy": ["last", "mean", "drift"]}
gscv = ForecastingGridSearchCV(forecaster=forecaster, param_grid=param_grid, cv=cv)
gscv.fit(y)
# %%
y_pred = gscv.predict(fh)

y = load_shampoo_sales()
pipe = TransformedTargetForecaster(
    steps=[("imputer", Imputer()), ("forecaster", NaiveForecaster())]
)
cv = ExpandingWindowSplitter(
    initial_window=24, step_length=12, start_with_window=True, fh=[1, 2, 3]
)
gscv = ForecastingGridSearchCV(
    forecaster=pipe,
    param_grid=[
        {
            "forecaster": [NaiveForecaster(sp=12)],
            "forecaster__strategy": ["drift", "last", "mean"],
        },
        {
            "imputer__method": ["mean", "drift"],
            "forecaster": [ThetaForecaster(sp=12)],
        },
        {
            "imputer__method": ["mean", "last"],
            "forecaster": [ExponentialSmoothing(sp=12)],
            "forecaster__trend": ["add", "mul"],
        },
    ],
    cv=cv,
    n_jobs=-1,
)
gscv.fit(y)
y_pred = gscv.predict(fh=[1, 2, 3])
