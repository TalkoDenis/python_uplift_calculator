import statistics
from typing import List, Optional
from src.exceptions import DataValidationError

class DataPreprocessor:

    @staticmethod
    def _validate_imputation_input(values: List[Optional[float]]) -> None:
        
        if not values:
            return

        if all(v is None for v in values):
            raise DataValidationError('All values are None!')
        
        if values [0] is None or values[-1] is None:
            raise DataValidationError('The first and the last ones value cannot be empty')

    @staticmethod
    def impute_missing_values(values: List[Optional[float]], strategy: str='linear') -> List[float]:

        if not values:
            return []

        DataPreprocessor._validate_imputation_input(values)
        
        if strategy == 'zero':
            return DataPreprocessor._impute_zero(values)
        elif strategy == 'forward_fill':
            return DataPreprocessor._impute_forward_fill(values)
        elif strategy == 'linear':
            return DataPreprocessor._impute_linear(values)
        elif strategy == 'mean':
            return DataPreprocessor._impute_mean(values)
        elif strategy == 'median':
            return DataPreprocessor._impute_median(values)
        else:
            raise ValueError(f'The wrong strategy {strategy}')

    @staticmethod
    def _impute_zero(values: List[Optional[float]]) -> List[float]:
        return [0.0 if v is None else v for v in values]

    def _impute_forward_fill(values: List[Optional[float]]) -> List[float]:
        result = []
        last_valid = values[0]
        for v in values:
            if v is not None:
                last_valid = v
            result.append(last_valid)
        return result

    @staticmethod
    def _impute_linear(values: List[Optional[float]]) -> List[float]:
        result = list(values)
        n = len(result)

        i = 0
        while i < n:
            if result[i] is None:
                j = i + 1
                while j < n and result[j] is None:
                    j += 1
                start_val = result[i - 1]
                end_val = result[j]
                steps = j - (i - 1)

                step_size = (end_val - start_val) / steps

                for k in range(i, j):
                    multiplier = k - (i - 1)
                    result[k] = start_val + step_size * multiplier
                i = j
            else:
                i += 1
        return result

    @staticmethod
    def _impute_mean(values: List[Optional[float]]) -> List[float]:
        valid_numbers = [v for v in values if v is not None]
        avg_values = statistics.mean(valid_numbers)
        return [avg_values if v is None else v for v in values]

    @staticmethod
    def _impute_median(values: List[Optional[float]]) -> List[float]:
        valid_numbers = [v for v in values if v is not None]
        avg_values = statistics.median(valid_numbers)
        return [avg_values if v is None else v for v in values]
