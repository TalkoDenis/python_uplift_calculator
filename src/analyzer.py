import datetime
from typing import List, Tuple, Dict, Any, Optional
from src.base import UpliftResult, BaseForecaster
from src.data import TimeSeriesData
from src.preprocessor import DataPreprocessor
from src.models import LinearTrendForecaster
from exceptions import InsufficientDataError, DataValidationError

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
            raise InsufficientDataError('Do not have data!')
