import statistics
from typing import List, Optional, Tuple
from src.base import BaseForecaster
from src.exceptions import InsufficientDataError, NotFittedError

class LinearTrendForecaster(BaseForecaster):
    def __init__(self, floor_value: Optional[float] = 0.0):
        self.floor_value = floor_value
        self.alpha: Optional[float] = None
        self.beta: Optional[float] = None

    @staticmethod    
    def _calculate_coefficients(x: List[int], y: list[float]) -> Tuple[float, float]:
        n = len(x)
        if n < 2:
            raise InsufficientDataError("Need more points")

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numeration = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator = sum((xi - mean_x) **2 for xi in x)

        if denominator == 0:
            beta = 0.0
        else:
            beta = numeration / denominator

        alpha = mean_y - beta * mean_x

        return alpha, beta

    def fit(self, x: List[int], y: List[float]) -> None:
        alpha, beta = self._calculate_coefficients(x, y)
        self.alpha = alpha
        self.beta = beta


    def predict(self, x: List[int]) -> List[float]:
        if self.alpha is None or self.beta is None:
            raise NotFittedError("The model is not fited!")

        predictions = []
        for xi in x:
            pred_y = self.alpha + self.beta * xi
            if self.floor_value is not None and pred_y < self.floor_value:
                pred_y = self.floor_value
            predictions.append(pred_y)
        return predictions

class Averege_forecaster(BaseForecaster):
    def __init__(self, floor_value: Optional[float] = 0.0):
        self.floor_value = floor_value
        self.mean_value: Optional[float] = None

    def fit(self, x: List[int], y: List[float]) -> None:
        if len(y) < 1:
            raise InsufficientDataError('For calculate the maen value need more then one point')
        self.mean_value = statistics.mean(y)

    def predict(self, x: List[int]) -> List[float]:
        if self.mean_value is None:
            raise NotFittedError('The model is not fitted!')
        
        predictions = []
        for _ in x:
            pred_y = self.mean_value
            if self.floor_value is None and pred_y < self.floor_value:
                pred_y = self.floor_value
            predictions.append(pred_y)
        return predictions

class NaiveForecuster(BaseForecaster):
    def __init__(self, floor_value: Optional[float] = 0.0) -> None:
        self.floor_value = floor_value
        self.last_value: Optional[float] = None
    
    def fit(self, x: List[int], y: List[float]) -> None:
        if len(y) < 1:
            raise InsufficientDataError('For calculate the maen value need more then one point')
        self.last_value = y[-1]

    def predict(self, x: List[int]) -> List[float]:
        if self.last_value is None:
            raise NotFittedError('The model is not fitted!')
        
        predictions = []
        for _ in x:
            pred_y = self.last_value
            if self.floor_value is None and pred_y < self.floor_value:
                pred_y = self.floor_value
            predictions.append(pred_y)
        return predictions
