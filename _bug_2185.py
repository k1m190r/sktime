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
        {"imputer__method": ["mean", "drift"], "forecaster": [ThetaForecaster(sp=12)],},
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


# %%
import numpy as np
from sktime.datasets import load_shampoo_sales
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_evaluation._functions import _split

y = load_shampoo_sales()
X = None
train = np.arange(24)
test = np.array([24, 25, 26])
fh = [1, 2, 3]
fit_params = {}

forecaster = ExponentialSmoothing(trend="mul", sp=12)
y_train, y_test, X_train, X_test = _split(y, X, train, test, fh)
fh = ForecastingHorizon(y_test.index, is_relative=False)
forecaster.fit(y_train, X_train, fh=fh, **fit_params)
y_pred = forecaster.predict(fh, X=X_test)
y_pred

# %%
