import datetime
from typing import List, Tuple, Dict, Any, Optional
from src.base import UpliftResult, BaseForecaster
from src.data import TimeSeriesData
from src.preprocessor import DataPreprocessor
from src.models import LinearTrendForecaster
from src.exceptions import InsufficientDataError, DataValidationError

class UpliftAnalyzer:
    def __init__(self,
                 forecast_model: str='linear_trend',
                 missing_value_strategy: str='linear',
                 smoothing_strategy: Optional[str]=None,
                 windiw_size: int=3,
                 floor_value: Optional[float]=0.0,
                 allow_negative_uplift: bool=False):
        
        self.forecast_model = forecast_model
        self.missing_value_strategy = missing_value_strategy
        self.smoothing_strategy = smoothing_strategy
        self.windiw_size = windiw_size
        self.floor_value = floor_value
        self.allow_negative_uplift = allow_negative_uplift

        if self.forecast_model == 'linear_trend':
            self.model: BaseForecaster = LinearTrendForecaster(floor_value=self.floor_value)
        else:
            raise ValueError(f'Unnown model {self.forecast_model}')

    def analyze(self, raw_data: List[Tuple[datetime.date, float]], intervation_date: datetime.date) -> UpliftResult:
        ts_data = TimeSeriesData(raw_data)

        train_dates, train_y, test_dates, test_y, train_x, test_x = self._split_data(
            ts_data.dates, ts_data.values, intervation_date
        )

        if len(train_y) < 2:
            raise InsufficientDataError('Do not have enought data (need 2 or more points)')
        if self.smoothing_strategy and len(train_y) < self.windiw_size:
            raise InsufficientDataError(f'Train data is shorter then window {self.windiw_size}')
        if len(test_y) < 1:
            raise InsufficientDataError('Not enought data!')

        train_y_clean = DataPreprocessor.impute_missing_values(train_y, self.missing_value_strategy)
        test_t_clean = DataPreprocessor.impute_missing_values(test_y, self.missing_value_strategy)

        train_y_smooth = DataPreprocessor.smooth_data(train_y_clean, self.smoothing_strategy, self.windiw_size)

        self.model.fit( train_x, train_y_smooth)

        train_predictions = self.model.predict(train_x)
        train_mae = sum(abs(fact - pref) for fact, pred in zip(train_y_smooth, train_predictions)) / len(train_y_smooth)

        test_predictions = self.model.predict(test_x)

        total_uplift = 0.0
        for fact, pred in zip(test_y_clean, test_predictions):
            delta = fact - pred
            if not self.allow_negative_uplift and delta < 0:
                delta = 0.0
            total_uplift += delta

        sum_forecast = sum(test_predictions)
        if sum_forecast == 0:
            relative_uplift = 0.0
        else:
            relative_uplift = (total_uplift / sum_forecast) * 100.0

        return UpliftResult(
            total_uplift=total_uplift,
            relative_uplift_percentage=relative_uplift,
            train_mae=train_mae,
            test_dates=test_dates,
            forecast_data=test_predictions
        )

    @staticmethod
    def _split_data(
        dates: List[datetime.date],
        values: List[Optional[float]],
        intervention_data: datetime.date
    ) -> Tuple[List[datetime.date], List[Optional[float]], List[datetime.date], List[Optional[float]], List[int], List[int]]:
        if intervention_data not in dates:
            raise DataValidationError(f'Date {intervention_date} not in dates!')
        split_index = dates.index(intervention_date)
        
        train_dates = dates[:split_index]
        train_y = values[:split_index]
        train_x = list(range(0, split_index))

        test_dates = dates[split_index:]
        test_y = values[split_index:]
        test_x = list(range(split_index, len(dates)))

        return train_dates, train_y, test_dates, test_y, train_x, test_x
