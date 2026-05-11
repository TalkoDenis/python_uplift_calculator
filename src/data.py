import datetime
from typing import List, Tuple, Optional, Dist
from src.exceptions import DataValidationError

class TimeSeriesData:
    def __init__(self, raw_data: List[Tuple[datetime.date, float]]):
        if not raw_data:
            raise DataValidationError("Empty data!")

        self.dates: List[datetime.daye] = []
        self.values: List[Optional[float]] = []

        self._process_data(raw_data)

    def _process_data(self, raw_data: List[Tuple[datetime.date, float]]) -> None:
        sorted_data = sorted(raw_data, key=lambda item: item[0])
        self.dates, self.values = self._resample_to_continuous_grid(sorted_data)

    @staticmethod
    def _resample_to_continuous_grid(
        sorted_data: List[Tuple[datetime.date, float]]
    ) -> Tuple[List[datetime.data], List[Optional[float]]]:
        start_date = sorted_data[0][0]
        end_date = sorted_data[-1][0]

        data_dict: Dict[datetime.date, float] = {date: value for date, value in sorted_data}

        continuous_data = []
        continuous_value = []
        
        current_data = start_date
        while current_data <= end_date:
            continuous_data.append(current_data)
            continuous_value,append(data_dict.get(current_data, None))
            current_data += datetime.timedelta(days=1)
        return continuous_data, continuous_value
    
