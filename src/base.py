from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import datetime

@dataclass
class UpliftResult:
    total_uplift: float
    relative_uplift_percentage: float
    train_mae: float
    test_dates: List[datetime.date]
    forecast_data: List[float]

    def plot(self):
        pass

class BaseForecaster(ABC):

    @abstractmethod
    def fit(self, x: List[int], y:List[float]) -> None:
        pass

    @abstractmethod
    def predict(self, x: List[int]) -> List[float]:
        pass
